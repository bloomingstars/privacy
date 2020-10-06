import re, sys
import argparse
import json
from urllib.parse import urlparse
from tld import get_tld
from adblockparser import AdblockRules
from tabulate import tabulate

raw_rules = []
domains=list()
mimt={}
blocked_sites=set()
total_no_sites = 0
def read_files():
    easyList = "G:\Privacy\hw3\easylist.txt"
    fo = open(easyList, "r")
    r = 0
    for line in fo:
        chunks = re.split("\n", line)
        raw_rules.append(chunks[0])

def get_blocked_sites_domain(url):
    res = get_tld(url, as_object=True)
    maindomain=res.fld
    blocked_sites.add(maindomain)

def write_options():
    options={'third-party':False,'script':False,'image':False,'stylesheet':False,'object':False,'subdocument':False,'xmlhttprequest':False,'websocket':False,'webrtc':False,'popup':False,'generichide':False,'genericblock':False}
    if(mimt.get(dm) !=None and 'script' in mimt.get(dm) ):
        options['script']=True
    if(mimt.get(dm) !=None and 'css' in mimt.get(dm)):
        options['stylesheet']=True
    if(mimt.get(dm) !=None and 'image' in mimt.get(dm)):
        options['image']=True
    options['domain']= get_tld(dm, as_object=True).domain
    if dm not in current_site:
        options['third-party']= True
    return options
    

def main(harfile_path):
    #harfile = open('G:\Privacy\hw3\www.macys.com_Archive [19-10-11 16-36-30].har', 'r')
    harfile = open(harfile_path, encoding="utf8")
    harfile_json = json.loads(harfile.read())
    i = 0
    global total_no_sites
    for entry in harfile_json['log']['entries']:
        i = i + 1
        url = entry['request']['url']
        total_no_sites=total_no_sites+1
        domains.append(url)
        mimietype=entry['response']['content']
        if mimietype.get('mimeType')!=None:
            mimt[url]=mimietype.get('mimeType')

        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        prog='parsehar',
        description='Parse .har files into comma separated values (csv).')
    argparser.add_argument('harfile', type=str, nargs=2,
                        help='path to harfile to be processed.')
    args = argparser.parse_args()
    current_site=args.harfile[1]
    main(args.harfile[0])

read_files();
rules=AdblockRules(raw_rules,supported_options=['third-party','script','image','stylesheet','domain','object','subdocument','xmlhttprequest','websocket','webrtc','popup','generichide','genericblock'],skip_unsupported_rules=False)
print(rules)
no_sites_blocked=0
for dm in domains:
    options=write_options()
    if(rules.should_block(dm,options)):
        no_sites_blocked+=1
        get_blocked_sites_domain(dm)

print(tabulate([[ current_site, total_no_sites , no_sites_blocked,blocked_sites ]], headers=['Site'
,'# of total HTTP requests',
'# of HTTP requests blocked'
,'Third-party domains (not URL) blocked']))

