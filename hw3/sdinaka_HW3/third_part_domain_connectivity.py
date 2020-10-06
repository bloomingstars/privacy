import argparse
import json
from urllib.parse import urlparse
import networkx as nx
from tld import get_tld
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt

def main(harfile_path, cur_domain):
    g=nx.Graph()
    pos=nx.planar_layout(g)
    g.add_node(cur_domain)
    domains=set()
    #harfile = open('G:\Privacy\hw3\www.macys.com_Archive [19-10-11 16-36-30].har', 'r')
    harfile = open(harfile_path, encoding="utf8")
    harfile_json = json.loads(harfile.read())
    i = 0
    for entry in harfile_json['log']['entries']:
        i = i + 1
        url = entry['request']['url']
        urlparts = urlparse(url)
        res = get_tld(url, as_object=True)
        subdomain=res.subdomain
        maindomain=res.domain
        thisdomain= res.fld
        if maindomain != cur_domain and thisdomain not in domains and thisdomain !=cur_domain:
                print (thisdomain)
                domains.add(thisdomain)
            
    for dm in domains:
        if dm != cur_domain:
            g.add_node(dm,node_color='#FFFF33')
            e= g.add_edge(dm, cur_domain)
    nx.draw(g, with_labels=True, font_size=8)
    plt.show()
    print('')
    


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        prog='parsehar',
        description='Parse .har files into comma separated values (csv).')
    argparser.add_argument('harfile', type=str, nargs=2,
                        help='path to harfil to be processed.')
    args = argparser.parse_args()

    main(args.harfile[0], args.harfile[1])
    print('')
