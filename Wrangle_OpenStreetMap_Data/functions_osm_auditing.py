import xml.etree.cElementTree as ET
import re
from collections import defaultdict

def get_element(osm_file, tags=('node', 'way', 'relation')):
    '''
    get the element content for 'node', 'way', 'relation' tags
    
    Args:
        osm_file: osm file that need to be processed
        tags: name of tags that is being retreived content from

    Returns:
        content element of input tags

    '''
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

def audit_street_type(street_types, street_name):
    '''
    check if an input is in pre-defined street type list; 
    if not add the input street type into the list

    Args:
        street_types: a list of pre-definded street type names - stored in strings
        street_name: a string of street name that need to be audited
    '''
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    '''
    check if an input is a street name

    Arg:
        elem: an element object to be verified

    Returns:
        True if elem.attrib['k'] == "addr:street", False otherwise.
    '''
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    '''
    go through osm file to capture all the street names/types in it

    Arg:
        osmfile: osm file that need to be processed
    
    Returns:
        a set of street types and their respective value that are contained in the input osm
    '''
    osm_file = open(osmfile, "rb")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

mapping = { "St": "Street",
            "St.": "Street",
            "Rd.": "Road",
            "Ave": "Avenue",
            "Rd" : "Road"
            }

def update_mapping(mapping, names_found):
    '''
    add newly-found street types into the pre-defined mapping

    Args:
        mapping: a dictionary that defines which value needs to be converted to which value correspondingly 
        names_found: a list of names that need to be added to the mapping dictionary

    Returns:
        updated mapping dictionary with names in names_found being added to it

    '''
    for i in names_found:
        if i not in mapping:
            try: 
                int(i)
            except ValueError:
                mapping[i] = i
    return mapping


def update_name(name, mapping):
    '''
    update street names according to pre-defined mapping 

    Args:
        name: a string that need to be updated
        mapping: a dictionary that defines which value needs to be converted to which value correspondingly

    Returns:
        updated name, which is still a string

    '''
    split = name.split()
    try:
        int(split[-1][-1])
        try:
            mapping[split[-2]]
            split[-2] = mapping[split[-2]]
            name = ' '.join(split)
            pass
        except (KeyError, IndexError) as error:
            pass
    except ValueError:
        try:
            mapping[split[-1]]
            split[-1] = mapping[split[-1]]
            name = ' '.join(split)
            pass
        except KeyError:
            pass
    return name



def clean_osm(osmfile,mapping,filename):
    '''
    process osm/xml file by updating names stored in tags, and write back the updated name into file

    Args:
        osmfile: osm file that need to be processed
        mapping: a dictionary that defines which value needs to be converted to which value correspondingly
        filename: the output file name, in string format

    '''

    tree = ET.parse(osmfile)
    root = tree.getroot()

    for elem in root.findall('way'):
        for tag in elem.iter('tag'):
            if is_street_name(tag):
                tag.set('v', update_name(tag.attrib['v'], mapping))        
    tree.write('%s.xml' %filename, encoding='utf-8')

    for elem in root.findall('node'):
        for tag in elem.iter('tag'):
            if is_street_name(tag):
                tag.set('v', update_name(tag.attrib['v'], mapping))        
    tree.write('%s.xml' %filename, encoding='utf-8')

