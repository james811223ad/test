import pandas as pd, re, numpy as np, spacy
from multiprocessing import Pool
from calendar import month_name
from fuzzywuzzy.fuzz import ratio
import sqlite3
#test2
quick_mod = lambda x: x.replace(' ', r'[A-Za-z]*[\s\S]{,5}?[A-Za-z]*?')
quick_mod2 = lambda x: x.replace(' ', r'\S*[\s\S]{,5}?\S*?')

year_pat = r'(?:(?:18|19|20)[0-9]{2}|(?<=\D)[0-9]{2}(?=\D))'
date_pat = [
    r'\d{1,2}/\d{1,2}/', r'\d{1,2}-\d{1,2}-' ,
    r'[a-z]{3,}[-\s_./]{,2}\d{1,2}[-\s_.,/;]{1,4}',
    quick_mod(r'(?:\d{1,2}[a-z]{2}|[a-z]{3,}) day of [a-z]+ ,? ')
]
date_pat = '|'.join([d + year_pat for d in date_pat])
query_bmk = lambda df, bmk_pat: df.bmk.map(lambda x: bool(re.search(bmk_pat.replace(' ','.+'), str(x), re.I)))

def get_LS_labeled_data(sql_path = r'/home/yuwen/Data/label_studio.sqlite3'):
    '''
    Get data from LAbel Studio to be trained using Spacy 2.
    Parameters
    ----------
    sql_path : str, optional
        Path to the sqlite3 file for Label Studio.

    Returns
    -------
    list of training data ready for Spacy NER.
    '''
    con = sqlite3.connect(sql_path)#select id, data from task
    project = pd.read_sql_query('''select id, title from project
    ''', con)#where is_labeled = 1
    key = None
    if project.shape[0] > 1:
        print('Available projects:\n')
        for i, r in project.iterrows():
            print(f"Title: {r['title']}\tKey: {r['id']}\n")
        key = input('Enter project key to select.')
    query = '''
    select result, data, project_id from task_completion as tc
    join task t on t.id == tc.task_id
    '''
    if key != None: query += f"\nwhere project_id == {key}"
    df = pd.read_sql_query(query, con)
    
    def get_spacy_train(row):
        d = eval(row['data'])
        text = d[[x for x in d.keys() if x not in ['pdf_name', 'page_num', 'all_text', 'bmk']][0]]
        ents = []
        for data in eval(row['result']):
            data = data['value']
            ents += [(data['start'], data['end'], data['labels'][0])]
        return [(text, {'entities': ents})]

    return [x for x in df.apply(get_spacy_train, 1).sum() if x[0] != '']

def cue_text(cue_regex, text, lines_count, head = True, stop_regex = None):
    '''
    Parameters
    ----------
    cue_regex : re.Pattern
        The compiled re object with the pattern to search for.
    text : str
        The texts to extract the lines of texts.
    lines_count : int
        Lines of text to extract besides the line of text matching the pattern.
    head : bool, optional
        Whether to search for the first or last line of lines of text. The default is True.
    stop_regex : re.Pattern, optional
        Early stopping pattern. The default is None.

    Returns
    -------
    TYPE string
        Algorithm:
            1. Break up text by lines.
            2. Search each lines starting from top line.
            3. If the line match the pattern, return that line and several lines before or after it.
    '''
    lines = []
    all_lines = [t for t in text.splitlines() if re.search(r'\w\W+\w|\w{3,}', t)]
    stop = False
    for i,l in enumerate(all_lines):
        sl = l
        if not re.match(r'\S+', cue_regex.pattern):
            if i:
                t = re.findall(r'\S+', all_lines[i - 1])[-3:]
                sl = t[-1] + ' ' + sl if len(t) else sl

            try:
                sl = sl + ' ' + re.findall(r'\S+', all_lines[i + 1])[0]

            except:
                pass
            
        if cue_regex.search(sl) and len(lines) == 0:
            if head:
                lines += [l]
            else:
                lines = [ln for ln in all_lines[:i + 1]]
                return '\n'.join(lines[-lines_count - 1:])
                    
        elif len(lines):
            if (stop_regex != None) and stop_regex.search(l):
                stop = True
                
            else:
                lines += [l]
                        
        if (len(lines) > lines_count) or stop:
            break
        
    if len(lines): return '\n'.join(lines)

def get_list(file_path, upper = False):
    '''
    Get list of entities from a txt file, where entities are seperated by lines.
    Parameters
    ----------
    file_path : str
    upper : Bool, optional
        Use all upper case or not?

    Returns
    -------
    list
    '''
    with open(file_path) as f:
        lines = f.read().splitlines()
        if upper:
            return [s.upper() for s in lines if s != '']
        return [s for s in lines if s != '']

def table_info(sql_path = r'/home/ubuntu/.local/share/label-studio/label_studio.sqlite3'):
    '''
    Prints out all of the columns of every table in a sqlite3 DB.
    '''
    con = sqlite3.connect(sql_path)
    c = con.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for table_name in tables:
        table_name = table_name[0] # tables is a list of single item tuples
        table = pd.read_sql_query("SELECT * from {} LIMIT 0".format(table_name), con)
        print(table_name)
        for col in table.columns:
            print('\t-' + col)
        print()

def clean_month_name(s):
    '''
    Clean up month name using fuzzy matching.
    Parameters
    ----------
    s : str
        date string.

    Returns
    -------
    s : str
    '''
    if not re.search(r'[a-z]', s, re.I):
        return s
    #s = re.sub(r'[^-0-9A-z/\s,.]', '', s)
    s = ' '.join(re.findall(r'[0-9a-z,]+', s, re.I))
    s = re.sub(r'day\s*of', 'day of ', s, re.I)
    s = re.sub(r'\s+', ' ', s)
    months = re.findall(r'[a-z]+', s, re.I)
    month_names = [x[:5] for x in month_name[1:]]
    for m in [mn for mn in months if not ('day' == mn.lower())]:
        for i, mn in enumerate(month_names):
            score = ratio(m.lower()[:5], mn.lower())
            if score > 70:
                s = s.replace(m, month_name[i + 1].capitalize())
                break
    return s

def cln_date(match):
    d, m, y = match.group(1), match.group(2), match.group(3)
    if ratio(d.lower(), 'first') > 70:
        return m + ' 1, ' + y
    return match.group(0)

def strfdate(date, fm=r'%m-%d-%y'):
    '''
    Convert a date string to a specified format.
    Parameters
    ----------
    date : str
    fm : str, optional
        String format for date used in pandas. The default is %m-%d-%y.

    Returns
    -------
    str
        Formated date string or None, if it fails to convert.
    '''
    if type(date) != str:
        return None
    date = clean_month_name(date if type(date) == str else '')
    date = re.sub(quick_mod(r'([a-z]{3,}) a f ([a-z]{3,}) ([0-9]{4})'), cln_date, date, flags = re.I)
    if type(date) != str or len(re.findall(r'\d+', date)) < 2: return None
    if ',' in date:
        date = re.sub(r'\s*,\s*', ', ', date)

    if 'day' in date.lower():
        p = r'(\d{1,2})[a-z]{2,4}[-\s_.]+day.{,4}of.+?([a-z]{2,})\s?[^,]*,?[-\s_.]+(\d{4})'
    
        ret = re.findall(p, re.sub(r'[a-z]irs[a-z]', '1st', date, flags = re.I), re.I)

        if len(ret):
            d, m, y = ret[0]
            date = f'{m} {d}, {y}'
        else:
            return None

    try:
        return '*' + pd.to_datetime(date).strftime(fm)
    
    except:
        try:
            return '*' + pd.to_datetime(re.sub(r'\s*\.\s*', ' ', date)).strftime(fm)
        
        except:
            pass

def parallelize_dataframe(df, func, n_cores = 8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df