import requests
from trafilatura import bare_extraction
from bs4 import BeautifulSoup as bs
import Levenshtein as lev
import pandas as pd
from dateutil import parser
from dateutil.tz import tzutc
def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False

import ast
def find_image_caption(soup, image_url,threshold=25):
    '''
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    '''
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if not img_tag:
        return "Image not found"
    figure = img_tag.find_parent('figure')
    if figure:
        figcaption = figure.find('figcaption')
        if figcaption:
            return figcaption.get_text().strip()
    for sibling in img_tag.find_next_siblings(['div', 'p','small']):
        if sibling.get_text().strip():
            return sibling.get_text().strip()
    title = img_tag.get('title')
    if title:
        return title.strip()
    # Strategy 4: Use the alt attribute of the image
    alt_text = img_tag.get('alt')
    if alt_text:
        return alt_text.strip()

    return "Caption not found"

def load_ris_urls(path, topk=30):
    '''
    Load the URLs of Reverse Image Search results.
    '''
    with open(path) as file:
        urls = []
        raw_urls = []
        image_urls = []
        dataset_image_path = []
        f = file.read().split('\n')
        for i in range(len(f)):
            try:
                line =  f[i].split(' | ')[1].split(';')
                image_dict = ast.literal_eval(f[i].split(' | ')[2])
            except:
                pass
            for u in line[:10] :
                dataset_image_path.append(f[i].split(' | ')[0])
                raw_urls.append(u)
                if u in image_dict.keys():
                    image_urls.append(image_dict[u])
                else:
                    image_urls.append([])
                try:
                    urls.append(u.split('/')[0] + '//' + u.split('/')[2])
                except:
                    urls.append(u)
    return urls, raw_urls, image_urls, dataset_image_path

def is_fc_organization(url):
    '''
    Check that the evidence url does not come from a FC organization
    Note: the provided list does not include every single existing FC organization. Some FC articles might still pass through this filter.
    '''
    fc_domains = ['https://www.fastcheck.cl','https://pesacheck.org','https://africacheck.org','https://www.snopes.com',
            'https://newsmobile.in', 'https://211check.org', 'factcrescendo.com/', 'https://leadstories.com', 'https://www.sochfactcheck.com',
            'https://newschecker.in','https://www.altnews.in', 'https://dubawa.org', 'https://factcheck.afp.com', 'factly.in',
            'https://misbar.com/factcheck/', 'larepublica.pe/verificador/', 'fatabyyano.net/', 'https://www.vishvasnews.com/', "newsmeter.in" ,
            "boomlive", "politifact","youturn.in", "lemonde.fr/les-decodeurs","factuel.afp.com","thequint.com", "logicalindian.com/fact-check/",
            "timesofindia.com/times-fact-check", "indiatoday.in/fact-check/", "smhoaxslayer.com", "facthunt.in", "aajtak.in/fact-check/",
            "bhaskar.com/no-fake-news", "theprint.in/hoaxposed/", 'firstdraftnews.org']
    for d in fc_domains:
        if d in url:
            return True
    return False

def is_banned(url):
    '''
    Check if the evidence url is in the list of banned urls
    '''
    banned = [
        #Those websites are flagged as potential unsafe or block the webscraping process
        "legalaiddc-prod.qed42.net", "windows8theme.net", "darkroom.baltimoresun.com", "dn.meethk.com", "hotcore.info", "pre-inscription.supmti.ac.ma",
        "hideaways.in", "www.alhurra.com/search?st=articleEx", "anonup.com", "hiliventures", "middleagerealestateagent", "nonchalantslip.fr",
        "corseprestige.com", ".aa.com.tr",  "landing.rassan.ir", "aiohotzgirl.com", "hotzxgirl.com",
        #The content of those social media websites is not scrapable.
        "facebook.com", "twitter.c", "youtube.co", "linkedin.co", "tiktok.c", "quora.c", "gettyimages.", "reddit." ]
    for b in banned:
        if b in url:
            return True
    return False

def is_obfuscated_or_encoded(url):
    '''
    Check that the evidence url is not obfuscated or encoded.
    '''
    unquoted_url = url
    try:
        return '%' in unquoted_url or '//' in unquoted_url.split('/')[2]
    except:
        return True

def time_difference(date1, date2):
    '''
    Compute whether date1 preceeds date2
    '''
    # Parse the dates
    dt1 = parser.parse(date1)
    dt2 = parser.parse(date2)
    # Make both dates offset-aware, assuming UTC if no timezone is provided
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    return dt1 < dt2

def merge_data(evidence, evidence_metadata,dataset):
    '''
    Merge all evidence by dropping duplicates and applying 2 filters:
    1) The evidence is not the original FC article itself
    2) The evidence has been published before the FC article
    '''
    evidence_df = pd.DataFrame(evidence)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)
    dataset_df = pd.DataFrame(dataset)
    merged_data = pd.merge(evidence_df, evidence_metadata_df.drop_duplicates(subset='raw url')[['image path','raw url']].rename(columns={'raw url':'url'}), on='url',how='inner')
    merged_data = pd.merge(merged_data.rename(columns={'url':'evidence url'}),
                           dataset_df[['org','image path','publication date']].rename(columns={'publication date': 'date_filter'}),
                           on='image path',how='inner')
    merged_data  = merged_data.dropna(subset='evidence url')
    #Verify that the evidence is not the FC article itself.
    fc_mask = merged_data.apply(lambda row : False if row['org'] in row['evidence url'] or row['org'] in ''.join(row['image url']) else True, axis=1)
    merged_data = merged_data[fc_mask]
    #Remove evidence that have been published after the FC article or have no publication date
    merged_data = merged_data[~merged_data['date'].isnull()]
    time_mask = merged_data.apply(lambda row : True if time_difference(row['date'],row['date_filter']) else False,axis=1)
    merged_data = merged_data[time_mask]
    merged_data = merged_data[['image path','org','evidence url','title','author','hostname',
                           'description','sitename','date','image','image url','image caption']]
    merged_data = merged_data.drop_duplicates(subset=['evidence url','image path'])
    return merged_data

def merge_data_img(evidence, evidence_metadata, dataset):
    '''
    Merge all evidence by dropping duplicates and applying 2 filters:
    1) The evidence is not the original FC article itself
    2) The evidence has been published before the FC article
    '''
    evidence_df = pd.DataFrame(evidence)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)
    dataset_df = pd.DataFrame(dataset)
    merged_data = pd.merge(evidence_df, evidence_metadata_df.drop_duplicates(subset='raw url')[['image path','raw url']].rename(columns={'raw url':'url'}), on='url',how='inner')
    # merged_data.rename(columns={'url': 'evidence url'}, inplace=True)
    merged_data = pd.merge(merged_data.rename(columns={'url':'evidence url'}),
                           dataset_df[['org','claim', 'image_id', 'label']].rename(columns={'image_id': 'image path'}),
                           on='image path',how='inner')
    merged_data  = merged_data.dropna(subset='evidence url')
    #Verify that the evidence is not the FC article itself.
    fc_mask = merged_data.apply(lambda row : False if row['org'] in str(row['evidence url']) or row['org'] in str(row['image url']) else True, axis=1)
    merged_data = merged_data[fc_mask]
    #Remove evidence that have been published after the FC article or have no publication date
    merged_data = merged_data[~merged_data['date'].isnull()]
    # time_mask = merged_data.apply(lambda row : True if time_difference(row['date'],row['date_filter']) else False,axis=1)
    # merged_data = merged_data[time_mask]
    merged_data = merged_data[['image path','org','evidence url','title','author','hostname',
                           'description', 'text', 'sitename','date','image','image url','image caption']]
    merged_data = merged_data.drop_duplicates(subset=['evidence url','image path'])
    return merged_data

def merge_data_img(evidence, evidence_metadata, dataset):
    '''
    Merge all evidence by dropping duplicates and applying 2 filters:
    1) The evidence is not the original FC article itself
    2) The evidence has been published before the FC article
    '''
    evidence_df = pd.DataFrame(evidence)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)
    dataset_df = pd.DataFrame(dataset)
    merged_data = pd.merge(evidence_df, evidence_metadata_df.drop_duplicates(subset='raw url')[['raw url']].rename(columns={'raw url':'url'}), on='url',how='inner')
    # merged_data.rename(columns={'url': 'evidence url'}, inplace=True)
    merged_data = pd.merge(merged_data.rename(columns={'url':'evidence url'}),
                           dataset_df[['org','claim', 'image_id', 'label']].rename(columns={'image_id': 'image path'}),
                           on='image path',how='inner')
    merged_data  = merged_data.dropna(subset='evidence url')
    #Verify that the evidence is not the FC article itself.
    fc_mask = merged_data.apply(lambda row : False if row['org'] in str(row['evidence url']) or row['org'] in str(row['image url']) else True, axis=1)
    merged_data = merged_data[fc_mask]
    #Remove evidence that have been published after the FC article or have no publication date
    merged_data = merged_data[~merged_data['date'].isnull()]
    # time_mask = merged_data.apply(lambda row : True if time_difference(row['date'],row['date_filter']) else False,axis=1)
    # merged_data = merged_data[time_mask]
    merged_data = merged_data[['image path','org','evidence url','title','author','hostname',
                           'description', 'text', 'sitename','date','image','image url','image caption']]
    merged_data = merged_data.drop_duplicates(subset=['evidence url','image path'])
    return merged_data


def load_evi_urls(path):
    '''
    Load the URLs of DuckDuckGo Search results.
    '''
    with open(path) as file:
        claims = []
        urls = []
        raw_urls = []
        f = file.read().split('\n')
        for i in range(len(f)):
            try:
                line =  f[i].split(' | ')[1].split(';')
            except:
                pass
            for u in line[:10] :
                claims.append(f[i].split(' | ')[0])
                raw_urls.append(u)
                try:
                    urls.append(u.split('/')[0] + '//' + u.split('/')[2])
                except:
                    urls.append(u)

    return claims, urls, raw_urls

def get_filtered_retrieval_results(path, ris=True):
    '''
    Filter the results of reverse image search.
    '''
    if ris == True:
        urls, raw_urls, image_urls, dataset_image_path = load_ris_urls(path)
        retrieval_results = []
        # Iterate over the URLs and apply the conditions
        for i in range(len(urls)):
            # Create a dictionary for each set of URL data
            ris_data = {
                'image path': dataset_image_path[i],
                'domain url': urls[i],
                'raw url': raw_urls[i],
                'image urls': image_urls[i],
                'is_fc': is_fc_organization(urls[i]),
                'is_https': raw_urls[i].startswith('https')
            }
            # Apply additional conditions to each dictionary
            ris_data['is_banned'] = is_banned(ris_data['raw url'])
            ris_data['is_obfuscated'] = is_obfuscated_or_encoded(ris_data['raw url'])  # Assuming 'type' condition is handled elsewhere or removed
            ris_data['is_html'] = is_likely_html(ris_data['raw url'])
            # Selection condition
            ris_data['selection'] = ris_data['is_html'] and ris_data['is_https'] and not ris_data['is_obfuscated'] and not ris_data['is_banned']
            # Append the dictionary to the list if it meets all the criteria
            retrieval_results.append(ris_data)

    else:
        claims, urls, raw_urls = load_evi_urls(path)
        retrieval_results = []
        # Iterate over the URLs and apply the conditions
        for i in range(len(urls)):
            # Create a dictionary for each set of URL data
            evi_data = {
                "claim" : claims[i],
                'domain url': urls[i],
                'raw url': raw_urls[i],
                'is_fc': is_fc_organization(urls[i]),
                'is_https': raw_urls[i].startswith('https')
            }
            # Apply additional conditions to each dictionary
            evi_data['is_banned'] = is_banned(evi_data['raw url'])
            evi_data['is_obfuscated'] = is_obfuscated_or_encoded(evi_data['raw url'])  # Assuming 'type' condition is handled elsewhere or removed
            evi_data['is_html'] = is_likely_html(evi_data['raw url'])
            # Selection condition
            evi_data['selection'] = evi_data['is_html'] and evi_data['is_https'] and not evi_data['is_obfuscated'] and not evi_data['is_banned']
            # Append the dictionary to the list if it meets all the criteria
            retrieval_results.append(evi_data)

    # Filter the data based on the selection criteria
    selected_retrieval_results = [d for d in retrieval_results if d['selection']]

    return selected_retrieval_results
def is_likely_html(url):
    '''
    Check that the evidence url is html
    '''
    # List of common file extensions
    file_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.doc', '.docx', '.ppt', '.pptx', '.xls',
                       '.xlsx', '.txt', '.zip', '.rar', '.exe', '.svg', '.mp4', '.avi', '.mp3']

    # Extract the extension from the URL
    extension = '.' + url.rsplit('.', 1)[-1].lower()

    # Check if the URL ends with a common file extension
    if extension in file_extensions:
        return False
    else:
        return True

def extract_info_trafilatura(page_url,image_url=None):
    try:
        headers= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(page_url, headers=headers, timeout=(10,10))
        if response.status_code == 200:
            #Extract content with Trafilatura
            result = bare_extraction(response.text,
                                   include_images=True,
                                   include_tables=False)
            #Remove unnecessary contente
            keys_to_keep = ['title','author','url',
                            'hostname','description','sitename',
                            'date','text','language','image','pagetype']
            result = {key: result[key] for key in keys_to_keep if key in result}

            # Finding the image caption

            soup = bs(response.text, 'html.parser')
            if image_url:
                image_caption = []
                result['image url'] = image_url
                for img in image_url:
                    image_caption.append(find_image_caption(soup, img))
                image_caption.append(find_image_caption(soup,result['image']))
                result['image caption'] = image_caption
            result['url'] = page_url
            return result
        else:
            return "Failed to retrieve webpage"
    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    # url = 'https://www.shutterstock.com/search/car-parked-snow'
    # image_url = 'https://www.shutterstock.com/image-photo/chicago-il-usa-january-13-260nw-1285453522.jpg'
    #
    # res = extract_info_trafilatura(url, image_url)
    # print(res)

    ll = 'https://www.snopes.com/author/arturo/;https://www.shutterstock.com/search/chicago-snow-storm;https://www.shutterstock.com/search/car-parked-snow;https://www.shutterstock.com/search/suv-car-snow?page=4;https://www.shutterstock.com/search/car-parked-badly?page=2;https://www.snopes.com/category/photos/?pagenum=49;https://www.snopes.com/fact-check/15-homeless-dead-in-chicago/;https://www.snopes.com/fact-check%20/rating/false/?pagenum=149;https://www.snopes.com/fact-check/?pagenum=494;https://www.snopes.com/fact-check/rating/false/?pagenum=149;https://www.shutterstock.com/es/search/snow-in-chicago;https://www.shutterstock.com/es/search/chicago-row-houses?page=2'
    print(len(ll.split(";")))

