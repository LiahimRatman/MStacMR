from collections import Counter
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from dao import load_from_json
from Faster_RCNN_train.class_mapa import attr_mapa, obj_mapa_my, obj_, attr


PATH = "C:/Users/Mikhail Korotkov/PycharmProjects/MStacMR/"


class_mapa = load_from_json(PATH + "checkpoints_and_vocabs/class_mapa_fin.json")
class_mapa_attr = load_from_json(PATH + "checkpoints_and_vocabs/class_mapa_attr.json")
json_map = load_from_json(PATH + "checkpoints_and_vocabs/image_data_clean.json")
# obj_map = load_from_json(PATH + "checkpoints_and_vocabs/objects.json")
attr_map = load_from_json(PATH + "checkpoints_and_vocabs/attributes.json")

full_data = []
item_obj_list = []
item_attr_list = []
for img_data, attr_data in zip(json_map, attr_map):
    if len(attr_data['attributes']):
        for obj in attr_data['attributes']:
            if 'attributes' in obj:
                for name in obj['names']:
                    item_obj_list.append(name)
                for attribute in obj['attributes']:
                    item_attr_list.append(attribute.lower().strip().lstrip())
object_count_sorted = {item[0]: item[1] for item in
                       sorted(dict(Counter(item_obj_list)).items(), key=lambda x: (-x[1], x[0]), reverse=False)}
attribute_count_sorted = {item[0]: item[1] for item in
                          sorted(dict(Counter(item_attr_list)).items(), key=lambda x: (-x[1], x[0]), reverse=False)}


TOP_CLASSES_COUNT = 1023
TOP_ATTR_CLASSES_COUNT = 500
k_keys_sorted = {item[0]: item[1] for item in
                 sorted(object_count_sorted.items(), key=lambda x: x[1], reverse=True)[:TOP_CLASSES_COUNT]}
k_keys_sorted_attr = {item[0]: item[1] for item in
                      sorted(attribute_count_sorted.items(), key=lambda x: x[1], reverse=True)[:TOP_ATTR_CLASSES_COUNT]}
label_mapa_k = {item: _ for _, item in enumerate(k_keys_sorted.keys())}
label_mapa_attr_k = {item: _ for _, item in enumerate(k_keys_sorted_attr.keys())}
num_classes_k = len(label_mapa_k)
num_classes_attr_k = len(label_mapa_attr_k)
full_data_train = []
full_data_test = []
for img_data, attr_data in zip(json_map, attr_map):
    if len(attr_data['attributes']):
        attributes = []
        for item in attr_data['attributes']:
            if 'attributes' in item:
                for name in item['names']:
                    name_ = name
                    for name_key, item_names in obj_mapa_my.items():
                        if name in item_names:
                            name_ = name_key
                    if name_ in label_mapa_k and name_ not in obj_:
                        for attribute in item['attributes']:
                            attribute_ = attribute.lower().strip().lstrip()
                            for attribute_key, item_attributes in attr_mapa.items():
                                if attribute.lower().strip().lstrip() in item_attributes:
                                    attribute_ = attribute_key

                            if attribute_ in label_mapa_attr_k and attribute_ not in attr:
                                item_ = item.copy()
                                item_['attributes'] = [attribute_]
                                item_['names'] = [name_]
                                attributes.append(item_)
                            else:
                                item_ = item.copy()
                                item_['attributes'] = ["no_attr"]
                                item_['names'] = [name_]
                                attributes.append(item_)

        if attributes:
            if '/VG_100K_2/' in img_data['url']:
                url = img_data['url'].replace('https://cs.stanford.edu/people/rak248/', PATH + 'images2/')
            else:
                url = img_data['url'].replace('https://cs.stanford.edu/people/rak248/', PATH + 'images/')
            url = url.replace(PATH, '/content/VG/')
            add = {
                'image_id': img_data['image_id'],
                'width': img_data['width'],
                'height': img_data['height'],
                'url': url,
                'attributes': attributes
            }
            fl = 0
            for attribute in attributes:
                for name in attribute['names']:
                    if object_count_sorted[name] < 1000:
                        fl = 2
                    if object_count_sorted[name] < 3000:
                        fl = 1
            for _ in range(fl):
                full_data.append(add)
            full_data.append(add)

print(full_data[0], full_data[-1], len(full_data))

names = []
attrs = []
for item in full_data:
    for el in item['attributes']:
        for name in el['names']:
            names.append(name)
    for el in item['attributes']:
        for at in el['attributes']:
            attrs.append(at)

names = list(set(list(names)))
attrs = list(set(list(attrs)))


# attrs.remove("no_attr")
# class_mapa = {name: val for val, name in enumerate(names)}
# class_mapa_attr = {name: val for val, name in enumerate(attrs)}
TOP_CLASSES_COUNT_NEW = len(class_mapa)
TOP_ATTR_CLASSES_COUNT_NEW = len(class_mapa_attr)
# class_mapa_attr["no_attr"] = TOP_ATTR_CLASSES_COUNT_NEW - 1
print(TOP_CLASSES_COUNT_NEW, TOP_ATTR_CLASSES_COUNT_NEW)
print(class_mapa_attr)
print(class_mapa)


def get_vg_dicts(data, class_mapa, class_mapa_attr):
    dataset_dicts = []
    for idx, item in enumerate(data):
        record = {}

        filename = item['url']

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = item['height']
        record["width"] = item['width']

        objs = []
        for box in item['attributes']:
            obj = {
                "bbox": [box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_mapa[box['names'][0]],
                "attribute_id": class_mapa_attr[box['attributes'][0]]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


DatasetCatalog.register("vg_train_16", lambda key=None: get_vg_dicts(full_data, class_mapa, class_mapa_attr))
MetadataCatalog.get("vg_train_16").set(thing_classes=list(class_mapa.keys()), thing_classes_attr=list(class_mapa_attr.keys()))

