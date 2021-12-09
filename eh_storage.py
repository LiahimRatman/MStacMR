import json
import embeddinghub as eh
from dao import load_from_json
import uuid
import numpy as np


# Saving CTC to hub
ctc_map = load_from_json('checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json')
data = {}
data_set = {}
img_emb = np.load('saved_embs_CTC.npy')
for item, emb in (ctc_map, img_emb):
    image_sp = item['image_path'].split('/')
    image_name = image_sp[-1]#.lower()
    image_source = image_sp[0].lower()
    image_uuid = str(uuid.uuid4())
    data[image_uuid] = {
        'image_name': image_name,
        'image_source': image_source,
        'image_uuid': image_uuid,
        'image_url': 'http://images.cocodataset.org/train2014/' + image_name,
    }
    data_set[image_uuid] = emb


with open('CTC_image_name_mapa.json', 'w') as f:
    f.write(json.dumps(data))

hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
space = hub.create_space("ctc_image_embs", dims=512)
space.multiset(data_set)


# import embeddinghub as eh
#
#
# # hub = eh.connect(eh.LocalConfig("eh_data/"))
# hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
# space = hub.create_space("quickstart", dims=3)
# embeddings = {
#     "apple": [1, 0, 0],
#     "orange": [1, 1, 0],
#     "potato": [0, 1, 0],
#     "chicken": [-1, -1, 0],
# }
# space.multiset(embeddings)
# # hub.save()
#
# space = hub.get_space("quickstart")
# neighbors = space.nearest_neighbors(key="apple", num=2)
# # print(neighbors)
#
