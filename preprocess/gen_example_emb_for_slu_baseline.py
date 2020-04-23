
from src.slu.datareader import datareader
from src.utils import load_embedding_from_npy
import numpy as np
import pickle

# domain to slot
domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

# use two examples
slot2example = {
    # AddToPlaylist
    "music_item": ["song", "track"],
    "playlist_owner": ["my", "donna s"],
    "entity_name": ["the crabfish", "natasha"],
    "playlist": ["quiero playlist", "workday lounge"],
    "artist": ["lady bunny", "lisa dalbello"],
    # BookRestaurant
    "city": ["north lima", "falmouth"],
    "facility": ["smoking room", "indoor"],
    "timeRange": ["9 am", "january the twentieth"],
    "restaurant_name": ["the maisonette", "robinson house"],
    "country": ["dominican republic", "togo"],
    "cuisine": ["ouzeri", "jewish"],
    "restaurant_type": ["tea house", "tavern"],
    "served_dish": ["wings", "cheese fries"],
    "party_size_number": ["seven", "one"],
    "poi": ["east brady", "fairview"],
    "sort": ["top-rated", "highly rated"], 
    "spatial_relation": ["close", "faraway"],
    "state": ["sc", "ut"],
    "party_size_description": ["me and angeline", "my colleague and i"],
    # GetWeather
    "current_location": ["current spot", "here"],
    "geographic_poi": ["bashkirsky nature reserve", "narew national park"],
    "condition_temperature": ["chillier", "hot"],
    "condition_description": ["humidity", "depression"],
    # PlayMusic
    "genre": ["techno", "pop"],
    "service": ["spotify", "groove shark"],
    "year": ["2005", "1993"],
    "album": ["allergic", "secrets on parade"],
    "track": ["in your eyes", "the wizard and i"],
    # RateBook
    "object_part_of_series_type": ["series", "saga"],
    "object_select": ["this", "current"],
    "rating_value": ["1", "four"],
    "object_name": ["american tabloid", "my beloved world"],
    "object_type": ["book", "novel"],
    "rating_unit": ["points", "stars"],
    "best_rating": ["6", "5"],
    # SearchCreativeWork
    # SearchScreeningEvent
    "movie_type": ["animated movies", "films"],
    "object_location_type": ["movie theatre", "cinema"],
    "location_name": ["amc theaters", "wanda group"],
    "movie_name": ["on the beat", "for lovers only"]
}

def gen_example_embs_based_on_each_domain(emb_file):
    # 1. generate example embeddings for each slot
    # get vocabulary
    _, vocab = datareader()
    # get word embeddings
    embedding = load_embedding_from_npy(emb_file)

    example2embs = {}
    for slot, examples in slot2example.items():
        example_embs_list = []
        for example in examples:
            tok_list = example.split()
            emb = np.zeros(400)  # word and char level embeddings
            for token in tok_list:
                index = vocab.word2index[token]
                emb = emb + embedding[index]
            example_embs_list.append(emb)
        
        example2embs[slot] = np.stack(example_embs_list, axis=-1)

    # 2. generate example embeddings based on each domain
    example_embs_based_on_each_domain = {}
    for domain, slot_names in domain2slot.items():
        example_embs = np.zeros((len(slot_names), 400, 2))
        for i, slot in enumerate(slot_names):
            embs = example2embs[slot]
            example_embs[i] = embs
        
        example_embs_based_on_each_domain[domain] = example_embs
    
    with open("../data/snips/emb/example_embs_based_on_each_domain.dict", "wb") as f:
        pickle.dump(example_embs_based_on_each_domain, f)

if __name__ == "__main__":
    gen_example_embs_based_on_each_domain("../data/snips/emb/slu_word_char_embs.npy")
