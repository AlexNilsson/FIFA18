import csv
from itertools import groupby

def csv_to_dict(csv_file):
  data = []
  
  with open(csv_file) as fin:
    reader = csv.reader(fin, skipinitialspace=True, quotechar="'")

    for keys in reader: break

    for row in reader:
      data.append(dict(zip(keys, row)))

  return data


def keep_entries(list_of_dicts, keys_to_keep):
  out = []
  for row in list_of_dicts:
    out.append({
      key: row[key] for key in keys_to_keep
    })
  return out

def group_by_key(list_of_dicts, key):
  out = {}
  for k, values in groupby(list_of_dicts, key=lambda x:x[key]):
    out[k] = [list(x.values())[1:] for x in values]
  return out

def unique_members_from_columns(list_of_dicts, keys):
  out = []
  for row in list_of_dicts:
    for key in keys:
      out.append(row[key])
  return list(set(out))