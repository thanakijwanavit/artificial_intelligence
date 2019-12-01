import pickle
import pprint

with open( 'data.pickle' , 'rb') as f:
    book = pickle.load(f)
#pprint.pprint(book)
print(len(book))