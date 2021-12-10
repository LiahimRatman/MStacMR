# import json
# import embeddinghub as eh
# from dao import load_from_json
# import uuid
# import numpy as np
#
#
# ctc_map = load_from_json('checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json')
# data = {}
# data_set = {}
# img_emb = np.load('saved_embs_CTC.npy')
# hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
# space = hub.create_space("ctc_image_embs5", dims=512)
# for item, emb in zip(ctc_map, img_emb):
#     image_sp = item['image_path'].split('/')
#     image_name = image_sp[-1]
#     image_source = image_sp[0].lower()
#     image_uuid = str(uuid.uuid4())
#     data[image_uuid] = {
#         'image_name': image_name,
#         'image_source': image_source,
#         'image_uuid': image_uuid,
#         'image_url': 'http://images.cocodataset.org/train2014/' + image_name,
#     }
#     # space.set(key=image_uuid, embedding=[float(item) for item in emb])
#     print(image_uuid)
#     data_set[image_uuid] = [float(item) for item in emb]
#
#
# space.multiset(data_set)
#
# with open('CTC_image_name_mapa_new.json', 'w') as f:
#     f.write(json.dumps(data))
#
#
# print([float(item) for item in emb])
# space.set("test", [float(item) for item in emb])
# neighbors = space.nearest_neighbors(key="test", num=2)
# print(neighbors)
# space.multidelete(["test"])
#
#
# # # print(data_set)
# #
# # print("DONE")
# # print(data[list(data.keys())[0]]['image_name'])
# # neighbors = space.nearest_neighbors(key=list(data.keys())[0], num=2)
# # print(neighbors)
# # for neighbor in neighbors:
# #     print(data[neighbor]['image_name'])
# #
# # # import embeddinghub as eh
# # #
# # #
# # # # hub = eh.connect(eh.LocalConfig("eh_data/"))
# # # hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
# # # space = hub.create_space("quickstart", dims=3)
# # embeddings = {
# #     "apple": [1., 0., 0.],
# #     "orange": [1., 1., 0.],
# #     "potato": [0., 1., 0.],
# #     "chicken": [-1., -1., 0.],
# # }
# # space.multiset(embeddings)
# # # # hub.save()
# # #
# # # space = hub.get_space("ctc_image_embs3")
# # # neighbors = space.nearest_neighbors(key="test", num=2)
# # # # print(neighbors)
# # #
# # # space.set(key="611fe120-0ba5-4911-8609-0c96abfb83af", embedding=[float(1) for _ in range(512)])
# # import embeddinghub as eh
# # hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
# # space = hub.get_space("ctc_image_embs1")
# # neighbors = space.nearest_neighbors(key='090230cf-f858-499d-9caf-76640129124e', num=3)
# # print(neighbors)
# # import json
# # with open('CTC_image_name_mapa.json', 'r') as f:
# #     data = json.load(f)
# #
# # print(data)
# # import embeddinghub as eh
# # import numpy as np
# # # np.ones(512, dtype=float)
# # print(np.ones(512, dtype=float))
# # eh.client.embedding_store_pb2.Embedding(np.ones(512, dtype=float))
# # space.set()
# # eh.client.embedding_store_pb2.Embedding([1., 1., 1.])
# # #
# # import embeddinghub as eh
# # hub = eh.connect(eh.Config(host="0.0.0.0", port=7462))
# # space = hub.create_space("ctc_image_embs2", dims=3)
# # space.get('test_caption2')
# # space.set(key='test_caption', embedding=[0., 0., 0.])
# # neighbors = space.nearest_neighbors(3, key='test_caption')
# # space.delete("test_caption2")
# # import numpy as np
# # print(type(np.array([1., 1., 1.])[0]))
# # space.download_snapshot()
#
#
#
#
#
# # import json
# # # import embeddinghub as eh
# # from dao import load_from_json
# # import uuid
# # import numpy as np
# #
# #
# # # Saving CTC to hub
# # ctc_map = load_from_json('checkpoints_and_vocabs/full_dataset_CTC_test_mapa_good.json')
# # data = {}
# # data_set = {}
# # img_emb = np.load('saved_embs_CTC.npy')
# # for item, emb in zip(ctc_map, img_emb):
# #     image_sp = item['image_path'].split('/')
# #     image_name = image_sp[-1]#.lower()
# #     image_source = image_sp[0].lower()
# #     image_uuid = str(uuid.uuid4())
# #     data[image_uuid] = {
# #         'image_name': image_name,
# #         'image_source': image_source,
# #         'image_uuid': image_uuid,
# #         'image_url': 'http://images.cocodataset.org/train2014/' + image_name,
# #     }
# #     data_set[image_uuid] = [item for item in emb]
# #
# #
# # with open('CTC_image_name_mapa.json', 'w') as f:
# #     f.write(json.dumps(data))
# #
# # print(type([float(item) for item in emb][0]))
# #
# #
# #
# # from vektonn import Vektonn
# # from vektonn.dtos import AttributeDto, AttributeValueDto, InputDataPointDto, VectorDto, SearchQueryDto
# #
# #
# # vektonn_client = Vektonn('http://localhost:8081')
# #
# #
# # vektonn_client.upload(
# #     data_source_name='QuickStart.Source',
# #     data_source_version='1.0',
# #     input_data_points=[
# #         InputDataPointDto(
# #             attributes=[
# #                 AttributeDto(key='id', value=AttributeValueDto(int64=1)),
# #                 AttributeDto(key='payload', value=AttributeValueDto(string='sample data point')),
# #             ],
# #             vector=VectorDto(is_sparse=False, coordinates=[3.14, 2.71]))
# #     ])
# #
# #
# # k = 10
# # query_vector = VectorDto(is_sparse=False, coordinates=[1.2, 3.4])
# #
# # search_results = vektonn_client.search(
# #     index_name='QuickStart.Index',
# #     index_version='1.0',
# #     search_query=SearchQueryDto(k=k, query_vectors=[query_vector]))
# #
# # print(f'For query vector {query_vector.coordinates} {k} nearest data points are:')
# # for fdp in search_results[0].nearest_data_points:
# #     attrs = {x.key: x.value for x in fdp.attributes}
# #     distance, vector, dp_id, payload = fdp.distance, fdp.vector, attrs['id'].int64, attrs['payload'].string
# #     print(f' - "{payload}" with id = {dp_id}, vector = {vector.coordinates}, distance = {distance}')
#
#
# # import numpy as np
# # import faiss                   # make faiss available
# # d = 64                           # dimension
# # index = faiss.IndexFlatL2(d)   # build the index
# #
# #
# # nb = 100000                      # database size
# # nq = 10000                       # nb of queries
# # np.random.seed(1234)             # make reproducible
# # xb = np.random.random((nb, d)).astype('float32')
# # xb[:, 0] += np.arange(nb) / 1000.
# # xq = np.random.random((nq, d)).astype('float32')
# # xq[:, 0] += np.arange(nq) / 1000.
# #
# # print(index.is_trained)
# # index.add(xb)                  # add vectors to the index
# # print(index.ntotal)
# #
# # k = 4                          # we want to see 4 nearest neighbors
# # D, I = index.search(xb[:5], k) # sanity check
# # print(I)
# # print(D)
# # D, I = index.search(xq, k)     # actual search
# # print(I[:5])                   # neighbors of the 5 first queries
# # print(I[-5:])                  # neighbors of the 5 last queries
#
# # from pymilvus import connections
# # connections.connect(alias="default", host='localhost', port='19530')
import numpy as np
em = np.load('saved_embs_CTC.npy')
print(em)
