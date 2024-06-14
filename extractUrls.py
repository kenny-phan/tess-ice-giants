import requests
import re
from bs4 import BeautifulSoup

def extractUrls(base_url, output_file):
    pattern = re.compile(r'https://archive\.stsci\.edu/missions/tess/ffi/s0070/2023/.*/1-1/tess2023.*-s0070-1-1-0265-s_ffic\.fits')

    visited_urls = set() 

    def fetch_urls_recursive(url):
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {base_url}. Status code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'lxml')
        visited_urls.add(url)
        subdirs = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('/') and not a['href'].startswith('../')]

        urls = []

        for subdir in subdirs:
            subdir_url = base_url.rstrip('/') + '/' + subdir.lstrip('/')
            if subdir_url not in visited_urls:
                visited_urls.add(subdir_url)
                urls.extend(fetch_urls_recursive(subdir_url))

        file_links = [a['href'] for a in soup.find_all('a', href=True)]
        for link in file_links: 
            full_url = base_url.rstrip('/') + '/' + link.lstrip('/')
            if pattern.match(full_url):
                urls.append(full_url)
        
        return urls

    urls_to_save = fetch_urls_recursive(base_url)


    with open(output_file, 'w') as f:
            for url in urls_to_save: 
                f.write(url + '/n')

    print(f"Extracted {len(urls_to_save)} URLs. Saved to {output_file}")

base_url = 'https://archive.stsci.edu/missions/tess/ffi/'
output_file = 'tess_urls.txt'

extractUrls(base_url, output_file)
