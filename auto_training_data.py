import os
import pickle
import json

# Directory of in your PC
DIRECTORY = os.getcwd()
os.remove(os.path.join(DIRECTORY, 'data/train2.json'))
os.remove(os.path.join(DIRECTORY, 'trained_documents.pickle'))
os.remove(os.path.join(DIRECTORY, 'trained_locations.pickle'))

# Function to find a new create file 
def absoluteFilePaths(directory):
    path = []
    files = []
    for dirpath, dirname, filenames in os.walk(directory):
        for f in filenames:
            if not os.path.basename(dirpath).startswith('.'):
                path.append(dirpath)
                files.append(f)
            
    return path, files

if os.path.exists(os.path.join(DIRECTORY, 'trained_documents.pickle')):
    trained_locations = pickle.load(open(os.path.join(DIRECTORY, 'trained_locations.pickle'), "rb"))
    trained_documents = pickle.load(open(os.path.join(DIRECTORY, 'trained_documents.pickle'), "rb"))
    trained_paths = [os.path.join(loc, doc) for loc, doc in zip(trained_locations, trained_documents)]
else:
    trained_locations = []
    trained_documents = []
    trained_paths = []

locations, documents = absoluteFilePaths(os.path.join(DIRECTORY, 'User'))
paths = [os.path.join(loc, doc) for loc, doc in zip(locations, documents)]

for doc in documents:
    if not doc in trained_documents:
        pickle.dump(locations, open(os.path.join(DIRECTORY, 'trained_locations.pickle'),"wb"))
        pickle.dump(documents, open(os.path.join(DIRECTORY, 'trained_documents.pickle'), "wb"))
        break;
        

# This code finds the new files and rewrites the whole training data
if os.path.exists(os.path.join(DIRECTORY, 'data','train2.json')):
    with open(os.path.join(DIRECTORY, 'data', 'train2.json')) as json_file:
        data = json.load(json_file)

else:
    data = {}
    data['intents'] = []



for loc, doc, path in zip(locations, documents, paths):
    if not path in trained_paths:
        data['intents'].append({
            "label": os.path.basename(loc) + '_' + str(doc).split('.')[0],                                                                   
            "train_data": [f"give me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report", 
                           f"open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report", 
                           f"show me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report", 
                           f"display {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report", 
                           f"please open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report",
                           f"give me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} document", 
                           f"open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} document", 
                           f"show me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} document", 
                           f"display {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} document", 
                           f"please open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} report",
                           f"give me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} file", 
                           f"open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} file", 
                           f"show me {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} file", 
                           f"display {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} file", 
                           f"please open {' '.join(''.join((char if char.isalpha() else ' ') for char in doc).split()[:-1])} {os.path.basename(loc)} file"],
            "responses": [os.path.join(loc, doc)],
            "context_set": ""
            })

        
with open(os.path.join(DIRECTORY, 'data', 'train2.json'), 'w') as outfile:
    json.dump(data, outfile)