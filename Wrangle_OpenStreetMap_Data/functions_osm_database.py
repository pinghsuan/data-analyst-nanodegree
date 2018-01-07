import codecs
import cerberus
import re
import csv
import pprint
import schema
import xml.etree.cElementTree as ET
import functions_osm_auditing as auditing

OSM_PATH = "processed_SanFrancisco.xml"
NODES_PATH = "SanFrancisco_nodes.csv"
NODE_TAGS_PATH = "SanFrancisco_nodes_tags.csv"
WAYS_PATH = "SanFrancisco_ways.csv"
WAY_NODES_PATH = "SanFrancisco_ways_nodes.csv"
WAY_TAGS_PATH = "SanFrancisco_ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

# get the element content for 'node', 'way', 'relation' tags
def get_element(osm_file, tags=('node', 'way', 'relation')):
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements
    
    ## fill node_attribs
    if element.tag == 'node':
        for key in NODE_FIELDS:
            if key in element.attrib:
                node_attribs[key] = element.attrib[key]
    ## fill node_tags
        for child in element:
            if child.tag == 'tag':
                tag_dic = {}
                tag_dic['id'] = node_attribs['id']
                tag_dic['key'] = child.attrib['k']
                tag_dic['value'] = child.attrib['v']
                k = str(child.attrib)[str(child.attrib).find(':')+1:str(child.attrib).find("',")+1]
                if k.find(':') > 0:
                    tag_dic['key'] = k[k.find(':')+1:-1]
                    tag_dic['type'] = k[2:k.find(':')]
                else:
                    tag_dic['type'] = 'regular'
                tags.append(tag_dic)
    
    ## fill way_attribs
    if element.tag == 'way':
        position = 0
        for key in WAY_FIELDS:
            if key in element.attrib:
                way_attribs[key] = element.attrib[key]
        for child in element:
    ## fill way_nodes
            if child.tag == 'nd':
               nd_dic = {} 
               nd_dic['id'] = way_attribs['id']
               nd_dic['node_id'] = child.attrib['ref']
               nd_dic['position'] = position
               position += 1
               way_nodes.append(nd_dic)
    ## fill way_tags
            elif child.tag == 'tag':
                tag_dic = {}
                tag_dic['id'] = way_attribs['id']
                tag_dic['key'] = child.attrib['k']
                k = str(child.attrib)[str(child.attrib).find(':')+1:str(child.attrib).find("',")+1]
                if k.find(':') > 0:
                    tag_dic['key'] = k[k.find(':')+1:-1]
                    tag_dic['type'] = k[2:k.find(':')]
                else:
                    tag_dic['type'] = 'regular'
                tag_dic['value'] = child.attrib['v']
                tags.append(tag_dic)
    
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}



# ================================================== #
#               Helper Functions                     #
# ================================================== #

def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])
