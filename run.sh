#############################
####### Slot Filling ########
#############################

# coach
python slu_main.py --exp_name coach_lstmenc --exp_id pm_0 --bidirection --freeze_emb --tgt_dm PlayMusic --n_samples 0
python slu_main.py --exp_name coach_lstmenc --exp_id gw_50 --bidirection --freeze_emb --tgt_dm GetWeather --n_samples 50
python slu_main.py --exp_name coach_lstmenc --exp_id sse_20 --bidirection --freeze_emb --tgt_dm SearchScreeningEvent --n_samples 20
python slu_main.py --exp_name coach_lstmenc --exp_id rb_50 --bidirection --freeze_emb --tgt_dm RateBook --n_samples 50
python slu_main.py --exp_name coach_lstmenc --exp_id scw_0 --bidirection --freeze_emb --tgt_dm SearchCreativeWork --n_samples 0
python slu_main.py --exp_name coach_lstmenc --exp_id br_50 --bidirection --freeze_emb --tgt_dm BookRestaurant --n_samples 50
python slu_main.py --exp_name coach_lstmenc --exp_id atp_20 --bidirection --freeze_emb --tgt_dm AddToPlaylist --n_samples 20

# coach + template regularization
python slu_main.py --exp_name coach_tr_lstmenc --exp_id pm_50 --bidirection --freeze_emb --tr --tgt_dm PlayMusic --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id gw_50 --bidirection --freeze_emb --tr --tgt_dm GetWeather --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id sse_50 --bidirection --freeze_emb --tr --tgt_dm SearchScreeningEvent --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id rb_50 --bidirection --freeze_emb --tr --tgt_dm RateBook --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id scw_50 --bidirection --freeze_emb --tr --tgt_dm SearchCreativeWork --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id br_50 --bidirection --freeze_emb --tr --tgt_dm BookRestaurant --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
python slu_main.py --exp_name coach_tr_lstmenc --exp_id atp_50 --bidirection --freeze_emb --tr --tgt_dm AddToPlaylist --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50


# baseline model (CT)
python slu_baseline.py --exp_name ct --exp_id pm_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm PlayMusic --n_samples 0
python slu_baseline.py --exp_name ct --exp_id gw_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm GetWeather --n_samples 0
python slu_baseline.py --exp_name ct --exp_id sse_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm SearchScreeningEvent --n_samples 0
python slu_baseline.py --exp_name ct --exp_id rb_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm RateBook --n_samples 0
python slu_baseline.py --exp_name ct --exp_id scw_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm SearchCreativeWork --n_samples 50
python slu_baseline.py --exp_name ct --exp_id br_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm BookRestaurant --n_samples 0
python slu_baseline.py --exp_name ct --exp_id atp_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm AddToPlaylist --n_samples 0

# baseline model (RZT)
python slu_baseline.py --exp_name rzt --exp_id pm_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm PlayMusic --n_samples 0
python slu_baseline.py --exp_name rzt --exp_id gw_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm GetWeather --n_samples 0
python slu_baseline.py --exp_name rzt --exp_id sse_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm SearchScreeningEvent --n_samples 0
python slu_baseline.py --exp_name rzt --exp_id rb_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm RateBook --n_samples 0
python slu_baseline.py --exp_name rzt --exp_id scw_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm SearchCreativeWork --n_samples 50
python slu_baseline.py --exp_name rzt --exp_id br_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm BookRestaurant --n_samples 0
python slu_baseline.py --exp_name rzt --exp_id atp_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm AddToPlaylist --n_samples 0


# test coach (coach + TR) model on testset
python slu_test.py --model_path ./experiments/coach_lstmenc/pm_50/best_model.pth --model_type coach --n_samples 50 --tgt_dm PlayMusic
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/pm_50/best_model.pth --model_type coach --n_samples 50 --tgt_dm PlayMusic

# test baseline models on testset
python slu_test.py --model_path ./experiments/ct/pm_50/best_model.pth --model_type ct --n_samples 50 --tgt_dm PlayMusic
python slu_test.py --model_path ./experiments/rzt/pm_50/best_model.pth --model_type rzt --n_samples 50 --tgt_dm PlayMusic

# test coach model on seen and unseen slots
python slu_test.py --model_path ./experiments/coach_lstmenc/pm_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm PlayMusic --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/gw_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm GetWeather --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/sse_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm SearchScreeningEvent --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/rb_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm RateBook --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/scw_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm SearchCreativeWork --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/br_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm BookRestaurant --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_lstmenc/atp_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm AddToPlaylist --test_mode seen_unseen

# test coach + TR model on seen and unseen slots
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/pm_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm PlayMusic --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/gw_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm GetWeather --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/sse_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm SearchScreeningEvent --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/rb_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm RateBook --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/scw_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm SearchCreativeWork --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/br_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm BookRestaurant --test_mode seen_unseen
python slu_test.py --model_path ./experiments/coach_tr_lstmenc/atp_20/best_model.pth --model_type coach --n_samples 20 --tgt_dm AddToPlaylist --test_mode seen_unseen

# test baseline models on seen and unseen slots
# CT (seen and unseen slots)
python slu_test.py --model_path ./experiments/ct/pm_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm PlayMusic --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/gw_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm GetWeather --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/sse_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm SearchScreeningEvent --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/rb_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm RateBook --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/scw_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm SearchCreativeWork --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/br_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm BookRestaurant --test_mode seen_unseen
python slu_test.py --model_path ./experiments/ct/atp_20/best_model.pth --model_type ct --n_samples 20 --tgt_dm AddToPlaylist --test_mode seen_unseen

# RZT (seen and unseen slots)
python slu_test.py --model_path ./experiments/rzt/pm_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm PlayMusic --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/gw_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm GetWeather --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/sse_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm SearchScreeningEvent --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/rb_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm RateBook --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/scw_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm SearchCreativeWork --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/br_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm BookRestaurant --test_mode seen_unseen
python slu_test.py --model_path ./experiments/rzt/atp_20/best_model.pth --model_type rzt --n_samples 20 --tgt_dm AddToPlaylist --test_mode seen_unseen


### Ablation Study
# coach (using trs to encode entity tokens)
python slu_main.py --exp_name coach_trs --exp_id pm_0 --bidirection --freeze_emb --tgt_dm PlayMusic --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id gw_0 --bidirection --freeze_emb --tgt_dm GetWeather --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id sse_0 --bidirection --freeze_emb --tgt_dm SearchScreeningEvent --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id rb_0 --bidirection --freeze_emb --tgt_dm RateBook --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id scw_0 --bidirection --freeze_emb --tgt_dm SearchCreativeWork --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id br_0 --bidirection --freeze_emb --tgt_dm BookRestaurant --enc_type trs --n_samples 0
python slu_main.py --exp_name coach_trs --exp_id atp_0 --bidirection --freeze_emb --tgt_dm AddToPlaylist --enc_type trs --n_samples 0


# coach (sum entity token features)
python slu_main.py --exp_name coach_sum --exp_id pm_50 --bidirection --freeze_emb --tgt_dm PlayMusic --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id gw_50 --bidirection --freeze_emb --tgt_dm GetWeather --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id sse_50 --bidirection --freeze_emb --tgt_dm SearchScreeningEvent --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id rb_50 --bidirection --freeze_emb --tgt_dm RateBook --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id scw_50 --bidirection --freeze_emb --tgt_dm SearchCreativeWork --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id br_50 --bidirection --freeze_emb --tgt_dm BookRestaurant --enc_type none --n_samples 50
python slu_main.py --exp_name coach_sum --exp_id atp_50 --bidirection --freeze_emb --tgt_dm AddToPlaylist --enc_type none --n_samples 50



#############################
############ NER ############
#############################

### Zero-shot
# baseline BiLSTM-CRF
python ner_baseline.py --exp_name lstm --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --bilstmcrf

# baseline CT
python ner_baseline.py --exp_name ct --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4

# baseline RZT
python ner_baseline.py --exp_name rzt --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --hidden_dim 150 --use_example

# coach
python ner_main.py --exp_name coach_lstmenc --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4

# coach + tr
python ner_main.py --exp_name coach_tr_lstmenc --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --tr


### few-shot (20 samples and 50 samples)
# baseline BiLSTM-CRF
python ner_baseline.py --exp_name lstm --exp_id ner_20 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --bilstmcrf --n_samples 20
python ner_baseline.py --exp_name lstm --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --bilstmcrf --n_samples 50

# baseline CT
python ner_baseline.py --exp_name ct --exp_id ner_20 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --n_samples 20
python ner_baseline.py --exp_name ct --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --n_samples 50

# baseline RZT
python ner_baseline.py --exp_name rzt --exp_id ner_20 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --hidden_dim 150 --use_example --n_samples 20
python ner_baseline.py --exp_name rzt --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --hidden_dim 150 --use_example --n_samples 50

# coach
python ner_main.py --exp_name coach_lstmenc --exp_id ner_20 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --n_samples 20
python ner_main.py --exp_name coach_lstmenc --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --n_samples 50

# coach + tr
python ner_main.py --exp_name coach_tr_lstmenc --exp_id ner_20 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --tr --n_samples 20
python ner_main.py --exp_name coach_tr_lstmenc --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --tr --n_samples 50


### Ablation Study
# coach (using trs to encode entity tokens)
python ner_main.py --exp_name coach_trs --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --enc_type trs
python ner_main.py --exp_name coach_trs --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4 --enc_type trs --n_samples 50

# coach (sum entity token features)
python ner_main.py --exp_name coach_sum --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --hidden_dim 150 --lr 1e-4 --enc_type none
python ner_main.py --exp_name coach_sum --exp_id ner_50 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --hidden_dim 150 --lr 1e-4 --enc_type none --n_samples 50
