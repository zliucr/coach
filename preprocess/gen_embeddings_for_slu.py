import numpy as np
import pickle

# prepare OoV words
oov_words = ['PAD', 'UNK', '50', '2016', 'indiespensables', 'biszewilo', '2010', '00', 'wastedagain', '6', 'palylist', '30', '52', 'twerkout', '2015', 'electronow', 'pre-party', 'a-hunting', '16', '61', '-blue', 'side-', 'teriazume', 'rapcaviar', '2', '2017', 'aftercluv', '2120', '170', '190', '60s', 'escapada', 'vyechnyy', '35', '1970', '1975', 'dcode2016', '100', '2005', 'elrow', '88', 'playlst', 'phunkadelic', '31', '70s', '20', 'friendesemana', 'godmusic', '1967', 'dschiwan', 'gasparjan', 'ain’t', '4', 'noctámbulo', 'electrosafari', '48', 'playslist', '80', '90s', '00s', 'trapeo', 'post-grunge', '90', 'rollerdisco', 'enamorándose', '80s', '2012', 'retrowave', 'i’m', 'indiespain', 'playlis', 'talkow', 'k-pop', '2008', 'l’ombre', '100%', '17', '157', '2009', '9th', '3', 'plalist', 'technical&brutal', '40', 'grigorjewna', 'semenowitsch', '54', 'tolmatschowa', 'hiptronix', '3rd', 'playlist:', '“chirping”', '74', '04', 'keltech', 'playlsit', 'lo-fi', 'rev-raptor', 'schalwowitsch', 'okudschawa', 'pachangueo', 'jarnowick', 'gougoush', '1994-2009', 'tsūzetsu', 'volume4', 'shadjarian', '9am', 'xsuie', 'plylist', 'szahram', '4813', '1959', '006', '2010s', 'savitsjeva', '21', '33', 'madrugando', 'radhae', 'aagathadi', 'm-cabi', 'bslade', '"jethro"', 'sci-fi', 'metalblood', '12', '2000', 'ljatoschynskyj', 'trapaholics', 'shangri-la', 'playist', 'oumarova', '59th', 'd-day', '1984', 'serving-men', '007', 'qriii', '10', '18', '23', '2023', '9', '1', '15', '7', '7th', 'close-by', 'top-rated', '22', '24th', '2027', 'searves', '8', '13th', '2037', 'restautant', '5', '18th', '2024', '20th', '22:54', '79', '55', '19:26', '27', '45', '2034', '4th', '2030', '7/27/2036', '13', '2026', '18:28', 'italian-american', '10th', '24', '2036', 'reestaurant', 'padangnese', 'fisn', 'urbanette', 'monty’s', '06:13', '1/1/2018', '17th', '15th', '14:41', 'eddie’s', 'reservea', 't-rex', 'osierfield', 'broadway-lafayette', '11th', 'twenty-eighth', '2018', 'twenty-sixth', 'lofgreen', 'smith-9th', '21:05:17', 'patetown', 'pidcoke', 'drive-in', '16th', '7/16/2027', 'co-operative', '04:36:28', 'spotat', '12th', 'parthenais-perrault', '2040', '13:22:34', '19', 'park-beach', '116th', '2028', '10/14/2026', 'cusinie', 'presskopf', 'rimsky-korsakoffee', '22nd', '14th', '105th', '0', '135th', '02:53', '2039', '2031', '2033', 'hitchland', '09:59', '72nd', 'ginestrata', '10:24', '02:22', '2020', '207th', '2112', 'jagual', '2019', '6th', 'fascility', 'festoni', '01:51:47', '2038', 'restarunt', '37', '2022', '21:49', 'giodi', '19:44:58', 'twenty-fourth', '17:38:04', '03:44', '10/22/2030', '141', 'reservtion', 'twenty-seventh', 'panisses', 'naytahwaush', '2021', 'twenty-second', '10:41:51', '10:47:15', 'resevation', 'pansette', 'vezione', '8th', 'restasurant', 'jean-georges', '2029', 'twenty-third', '09:58:27', '25', '2035', '05:51:52', '6/14/2035', 'guinea-bissau', '56', 'harry’s', '2/21/2021', '119', '3/21/2018', 'restaurnt', '2025', '345', 'kellerton', '18:49:20', '04:45', '163', 'trottole', 'av-69th', '11', 'milton-freewater', '4/17/2033', '103rd', '138th', '43', '11:16:07', 'av-barclays', 'fifty-five', 'baker’s', '11/1/2033', '6/1/2027', '08:05', 'maid-rite', 'pôchouse', '01:48:35', '20:38', 'week-end', '23rd', '2032', '00:55', 'i-lander', 'av-53rd', 'blvd-lehman', '1st', '8/8/2039', 'capicollo', 'bridge-city', '00:37', '7/25/2027', 'n9ne', '4/4/2036', '5/20/2028', '11/23/2031', '00:32', 'somewhee', '10/24/2028', 'restaurantin', '07:07', 'birthday/lincoln', '2/6/2020', '1/20/2023', 'cowansburg', '152', '28th', '5/20/2025', '06:42', '14', '42', 'servec', '26', '88th', 'st-boyd', '2nd', '13:22:09', 'sehlabathebe-nationalpark', 'layhigh', '27th', '333', '06:18:13', 'twenty-first', '28', '17:43', 'chatyrkul', '3/26/2023', '01:27', '05:00:34', '49', 'nellieburg', '00:00', 'wildreservaat', 'band-e', '3/22/2038', 'garrochales', '19:52', '10:21:20', 'tallahassee-st', 'solromar', '12/14/2023', '00:17', '02:45', '9/3/2034', 'dachigam-nationalpark', 'dovre-nationalpark', '4/19/2030', '8/4/2024', 'dochū-kōtsu', '22:23:22', '7/22/2030', 'forecase', '09:32:06', '9/11/2035', '02:02:30', '1/1/2031', 'runkaus', 'luambe-nationalpark', '09:30', '10/4/2021', 'tiplersville', 'meeres-nationalpark', '7/18/2030', '5th', 'twenty-fifth', '19th', '12/13/2025', '02:55:25', '12/10/2035', '5/17/2037', '213', '25th', 'le-aqua-na', 'maliau-basin-conservation-area', '06:59', '6/15/2025', 'altyn-emel-nationalpark', 'pointe-heath', '12/9/2039', 'rockholds', 'd’arguin', '12/28/2019', '09:04:38', 'nimule-nationalpark', '1/11/2030', 'granite-steppe', '8/26/2022', '131', 'atanassow-see', '02:59', 'cargray', '21st', '7/13/2036', '07:03:43', '188', 'maltio', '07:08:00', '11:47:52', '7/10/2023', '15:04', 'furano-ashibetsu', '12:06', '10/21/2024', 'lexington-fayette', 'kushiro-shitsugen', '1/11/2040', 'melcher-dallas', '165', '12/5/2032', '7/16/2032', '308', '07:08:02', '06:30:26', 'addo-elefanten-nationalpark', 'farristown', '10/2/2021', '6/21/2035', 'ak-suu', '2/21/2022', '175', '03:19:13', '26th', 'suthep-pui', '4/15/2034', '00:09:07', 'cairngorms-nationalpark', '05:44:13', 'foreast', '365', 'harrison–crawford', '09:42', '01:50', '03:43', '11:36:48', 'edesville', 'hahntown', '10:15', '07:43:21', '11/27/2023', '11/18/2018', '4/3/2027', 'forrecast', 'gipsy-gordon', 'glacier-nationalpark', '06:31:22', '32', '02:31', '129', 'favoretta', '257', 'camdeboo-nationalpark', '11:56', 'orlovista', '2/25/2025', '15:26:11', '11/12/2036', '13:19', '15:19:29', 'bothe-napa', '12/26/2018', 'tatra-nationalpark', '2/7/2021', '06:50:20', '06:05:48', '4/20/2038', 'alumb', 'top-20', 'zvooq', '2014', '2011', 'top-twenty', '1966', '1993', '1960', '1973', 'top-5', '1955', '1999', '1950', '1957', '1990', '1968', 'top-50', '1972', '1997', '1951', '1998', '2002', '1962', '1981', 'top-fifty', '2003', 'grebenchtchikov', '1954', '1956', '1958', 'top-five', '2004', '1994', 'chatschatrjan', 'yuauea', '1985', '1953', '1996', '2006', '1991', 'pop-folk', '1982', '1964', 'techno-industrial', 'burhøns', '1963', 'top-ten', '08', 'plpay', '2007', '1979', '1995', '1989', '1974', 'joo-hyun', '2gether', '2001', '1952', '1987', 'walerij', 'leontjew', 'sjedokowa', '1977', '1992', 'moutsouraev', '1969', 'akb48', '1965', 'is:', 'crazy=genius', '1988', '1986', '12"', '1976', 'bulanowa', 'top-10', '1980', 'cherry-tree', 'lyschytschko', 'nikolajewna', 'jessipowa', 'apbl98', 'plya', '1983', 'post-punk', 'trunes', 'pougatcheva', '1978', 'bedrossowitsch', 'kirkorow', 'vysotski', 'basjmet', '������', 'ohear', 'trip-hop', 'fun-punk', '1971', '2013', 'plkay', 'regulate…g', 'bachmet', 'ajinoam', 'yossif', 'streetdanz', 'guy-manuel', 'homem-christo', 'snuva', 'negerpunk', 'folk-rock', 'anatoljewitsch', 'kurjochin', 'u-roy', 'call:', 'e-type', 'sabbath:', 'britânico', 'van-pires', 'relaesd', 'mouse:', 'yet:', 'decoded:', 'freud:', 'oblivion:', 'trips:', 'history:', 'gota’s', 'kropp:', 'main-travelled', 'underwood:', 'reality:', 'paedophilia:', 'novels:', 'republic:', 'covering:', 'complex:', 'his-story', 'lullaby:', 'noise:', 'magic:', 'winnie-the-pooh', 'khaled:', 'fallout:', 'pornography:', 'obama:', 'flash:', 'ape-man', 'hindus:', 'tea-time', 'amityville:', 'tapshoes', '51', 'fox:', 'boys:', 'posiabble', 'islam:', 'broke:', 'ecology:', 'she-devil', 'ii:', 'benson-meditation', 'ice-cream', 'tar:', 'hundred-year', 'half-life', 'glamour:', '101:', 'wannabe:', 'beyonders:', 'piskies:', 'age:', 'pot-healer', 'trek:', 'duel:', 'rajinikanth:', 'tse-tung', '99', 'alchemyst:', 'a-z', 'wine-stained', 'notebook:', 'boone:', 'lee:', 'she:', 'hitler:', 'fall-down', 'pastwatch:', 'life:', 'mistborn:', 'drift:', 'stars:', 'emperors:', 'half-formed', '003½', 'confidence-man', 'man:', '1970–1980', 'fox-hunting', 'mandela:', 'phoenix:', 'chrome-plated', '1844', 'ledger:', 'maul:', 'rebbe:', 'terrorism:', 'scotia:', 'witch:', 'homicide:', 'superman:', 'willows:', '’72', 'shockscape', 'you:', '12:', 'that:', 'mundy:', 'halo:', '36', 'anti-japanese', '8-week', 'wilco:', 'gift:', 'theft:', 'temples:', 'gods:', 'cock-a-doodle-doo', '3:', 'playstation官方杂志', '64:', 'forget:', 'news-press', 'sun-star', 'i’ll', 'hundred-foot', 'seven-ups', 'f-1', '300:', 'a-myin-thit', 'rivalry:', 'wxhexeditor', 'show-ya', 'crisis:', 'live~legend', 'hits:', 'bloom:', 'zen:', 'taskcracker', 'cafe:', 'war:', 'center:', 'shell:', 'pursuit:', 'burnout:', '97:', 'iconic:', 'collection:', 'two-shoes', 'philosopher’s', 'phish:', 'black-body', 'they’re', 'karol:', 'in:', 'lou:', 'ice:', 'unreleased:', 'mailroom:', '180', '30th', 'dragons:', 'master:', 'ringin’', '1938', 'timerider:', 'ventura:', 'cd-rom', 'dante’s', 'i:', 'ochs:', 'warcraft:', 'pokémon:', 'crow:', 'awhile:', 'presents:', '4-hour', 'workin’', 'brenda’s', 'batman:', 'baby-sittor', 'lionheart:', '1634:', 'fimd', 'education:', 're:', 'jkt48', '1914–1918', 'beam:', 'junkies:', 'dark:', 'dots:', 'b-sides', 'hyun-joong', 'in-birth', 'sky:', 'duty:', 'friend:', 'supers:', 'frontiers:', 'mtv:', 'iii:', 'forever:', '2:', 'tron:', 'saint-pierre', 'rings:', 'sale:', 'love:', 'information:', 'don’t', 'crowd:', 'finiti', 'wide-eyed', 'reprise:', 'axis2', 'transformers:', 'boomtown:', 'dyskografia', 'chicago:', '1997-2003', 'el-palacio', '2007–2008', 'skyfall:', 'activity:', 'cosnarati:', 'frisbee:', 'gibraltar:', 'holla-day', 'iv:', 'fire:', 'heaven:', 'seasons:', 'butt-head', 'eyes:', 'storm:', 'babylon:', 'mafia:', 'pesterminator:', 'gadichindi', 'caledonian-record', 'time:', '911', 'gd&top', 'terrorists:', 'amrithavaahini', 'milagros:', 'times:', 'side:', 'baldur’s', 'champ:', '2k3', 'live:', 'well:', 'brunei:', 'hess:', 'peace-maker', 'tsuihou:', 'seven-thirty', 'clubbing:', 'winterheart’s', 'on-line', 'moon:', 'opa-opa', '101', 'flesh-colored', 'cash-cash', 'hellboy:', 'spartan:', 'perfume:', 'getaway:', 'countdown:', '7even', 'conan:', 'space:', '1945', 'of:', 'party’s', 'saga:', 'mary:', 'ethics:', 'whiskers:', 'nashville:', 'four:', 'knife-throwing', 'deal:50', 'bedroom:', 'best-of:', '2003–2013', 'canaich', 'elvis’', 'hip-hop', 'demi-gods', 'semi-devils', 'you’re', 'fresh:', 'wonderful:', 'changes:', 'bleach:', 'daimidaler:', 'hamilton:', 'conduct:', 'z:', 'think:', 'today:', 'north-west', 'movie:', 'space-age', 'tera:', 'cyrus:', 'mid-sixties', 'chavez:', 'battlestations:', 'sánchez:', '81', 'atla:', 'projekt:', 'v-the', '5:', 'show-biz', 'marilyn:', 'dolorosa:', 'corpus:', 'vi:', 'eve-olution', 'line:', 'work:', 'turtles:', 'cash:', 'drumline:', 'conker:', 'dale:', 'reproductions:', 'supernatural:', 'ball:', '12:26', '17:32:30', 'rhapsodies:', '78', 'lpaying', 'hendthighelbedi', 'alice:', '16:01:04', '04:34:15', 'sapheads', 'detective:', 'savage:', 'cream-trilogie', '03:01:48', 'amour:', '20:44', 'kingsman:', 'chuckys', 'star:', '1933-45', 'lady:', 'caseya', 'white:', 'days:', 'playinh', 'wynonna:', '11:09', 'xena:', 'beyblade:', 'nemiciamici', 'gaddar:', 'acchiappafantasmi', '07:31:32', 'fair:', 'ocean’s', 'em4jay', 'mabel’s', '04:08:11', 'sundown:', 'ennarukil', '21:41:08', 'bride’s', 'scheudle', 'morgan’s', 'entfesselt', 'vampate', 'there:', '02:39:23', 'daishūrai', 'todesfalle', '14:40', 'mulawin:', '10:37', '12:53', 'wars:', 'day:', 'hefner:', '11:12', '15:16:52', 'colic:', 'movietimes', '楽園追放', '-expelled', 'paradise-', '06:30', '00:47:43', 'fever:', '08:39', '1-1000', 'amor:', 'interview:', 'williams:', 'that’s', '07:25', 'titanic:', 'flow:', '10:56:18', 'anti-semitism', 'century:', 'salaam-e-ishq:', 'rock:', '01:19:00', '07:27', '15:02', '08:56:29', 'whaere', '807', 'plkaying', 'rangers:', 'kraken:', 'walt:', '20:45:24', 'g-men', 'peaks:', 'operation:', 'babar:', 'adventures:', 'ministers:', 'doa:', '16:', 'camping-car', '847:', '07:52', 'ganges:', '09:44']

slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description', 'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name', 'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city', 'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number', 'condition_description', 'condition_temperature']

# domain to description
domain2desp = {"AddToPlaylist": "add to playlist", "BookRestaurant": "reserve restaurant", "GetWeather": "get weather", "PlayMusic": "play music", "RateBook": "rate book", "SearchCreativeWork": "search creative work", "SearchScreeningEvent": "search screening event"}

# slot to description
slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'moive name', 'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}

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

def gen_oov_words():
    with open("../data/snips/emb/oov_words.txt", "w") as f:
        for oov in oov_words:
            f.write(oov + "\n")


def gen_embs_for_vocab():
    from src.slu.datareader import datareader
    from src.utils import load_embedding, init_experiment
    from config import get_params
    
    params = get_params()
    logger = init_experiment(params, logger_filename=params.logger_filename)

    _, vocab = datareader()
    embedding = load_embedding(vocab, 300, "PATH_OF_THE_WIKI_EN_VEC", "../data/snips/emb/oov_embs.txt")
    np.save("../data/snips/emb/slu_embs.npy", embedding)


def gen_slot_embs_based_on_each_domain(emb_file):
    ## 1. generate slot2embs
    slots = list(slot2desp.keys())
    desps = list(slot2desp.values())
    word2emb = {}
    # collect words
    for des in desps:
        splits = des.split()
        for word in splits:
            if word not in word2emb:
                word2emb[word] = []
    
    # load embeddings
    print("loading embeddings from %s" % emb_file)
    embedded_words = []
    with open(emb_file, "r") as ef:
        pre_trained = 0
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == 301
            except:
                continue
            if sp[0] in word2emb and sp[0] not in embedded_words:
                pre_trained += 1
                word2emb[sp[0]] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])
    print("Pre-train: %d / %d (%.2f)" % (pre_trained, len(list(word2emb.keys())), pre_trained / len(list(word2emb.keys()))))

    # calculate slot embs
    slot2embs = {}
    for i, slot in enumerate(slots):
        word_list = slot2desp[slot].split()
        embs = np.zeros(300)
        for word in word_list:
            embs = embs + word2emb[word]
        slot2embs[slot] = embs

    ## 2. generate slot2embs based on each domain
    slot_embs_based_on_each_domain = {}
    for domain, slot_names in domain2slot.items():
        slot_embs = np.zeros((len(slot_names), 300))
        for i, slot in enumerate(slot_names):
            embs = slot2embs[slot]
            slot_embs[i] = embs
        slot_embs_based_on_each_domain[domain] = slot_embs
    
    with open("../data/snips/emb/slot_embs_based_on_each_domain.dict", "wb") as f:
        pickle.dump(slot_embs_based_on_each_domain, f)


def combine_word_with_char_embs_for_slot(slotembs_dict_file):
    import torchtext
    char_ngram_model = torchtext.vocab.CharNGram()
    ## open wordlevel embeddings file
    with open(slotembs_dict_file, "rb") as f:
        slotembs_dict = pickle.load(f)
    word_char_embs_dict = {}
    for domain, slot_names in domain2slot.items():
        char_embs = np.zeros((len(slot_names), 100))
        for i, slot in enumerate(slot_names):
            embs = char_ngram_model[slot2desp[slot]].squeeze(0).numpy()
            char_embs[i] = embs
        word_embs = slotembs_dict[domain]
        word_char_embs_dict[domain] = np.concatenate((word_embs, char_embs), axis=-1)
    
    with open("../data/snips/emb/slot_word_char_embs_based_on_each_domain.dict", "wb") as f:
        pickle.dump(word_char_embs_dict, f)


def combine_word_with_char_embs_for_vocab(wordembs_file):
    from src.slu.datareader import datareader
    from src.utils import init_experiment
    from config import get_params
    import torchtext
    char_ngram_model = torchtext.vocab.CharNGram()

    params = get_params()
    logger = init_experiment(params, logger_filename=params.logger_filename)

    _, vocab = datareader()
    embedding = np.load(wordembs_file)

    word_char_embs = np.zeros((vocab.n_words, 400))
    for index, word in vocab.index2word.items():
        word_emb = embedding[index]
        char_emb = char_ngram_model[word].squeeze(0).numpy()
        word_char_embs[index] = np.concatenate((word_emb, char_emb), axis=-1)
    
    np.save("../data/snips/emb/slu_word_char_embs.npy", word_char_embs)


def add_slot_embs_to_slu_embs(slot_embs_file, slu_embs_file):
    from src.slu.datareader import datareader
    from src.utils import init_experiment
    from config import get_params

    with open(slot_embs_file, "rb") as f:
        slot_embs_dict = pickle.load(f)
    slu_embs = np.load(slu_embs_file)

    params = get_params()
    logger = init_experiment(params, logger_filename=params.logger_filename)
    _, vocab = datareader(use_label_encoder=True)

    new_slu_embs = np.zeros((vocab.n_words, 400))  # 400: word + char level embs

    # copy previous embeddings
    prev_length = len(slu_embs)
    new_slu_embs[:prev_length, :] = slu_embs

    for slot_name in slot_list:
        emb = None
        index = vocab.word2index[slot_name]
        if index < prev_length: continue
        for domain, slot_embs in slot_embs_dict.items():
            slot_list_based_on_domain = domain2slot[domain]
            if slot_name in slot_list_based_on_domain:
                slot_index = slot_list_based_on_domain.index(slot_name)
                emb = slot_embs[slot_index]
                break
        assert emb is not None
        new_slu_embs[index] = emb
    
    np.save("../data/snips/emb/slu_word_char_embs_with_slotembs.npy", new_slu_embs)

if __name__ == "__main__":
    # gen_oov_words()
    # gen_embs_for_vocab()
    # gen_domain_embs("PATH_OF_THE_WIKI_EN_VEC")
    # gen_slot_embs_based_on_each_domain("PATH_OF_THE_WIKI_EN_VEC")
    
    # combine_word_with_char_embs_for_slot("../data/snips/emb/slot_embs_based_on_each_domain.dict")
    # combine_word_with_char_embs_for_vocab("../data/snips/emb/slu_embs.npy")
    
    add_slot_embs_to_slu_embs("../data/snips/emb/slot_word_char_embs_based_on_each_domain.dict", "../data/snips/emb/slu_word_char_embs.npy")
