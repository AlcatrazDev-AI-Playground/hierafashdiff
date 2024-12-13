import os

model_root = os.path.join(os.path.dirname(__file__), "checkpoints")

dataset_root = "dataset/HieraFashion_5K/"

openpose_body_model_path = os.path.join(model_root, "body_pose_model.pth")
openpose_hand_model_path = os.path.join(model_root, "hand_pose_model.pth")

sam_model_path = os.path.join(model_root, "sam_vit_h_4b8939.pth")

model_yaml = os.path.join(os.path.dirname(__file__), "configs/cldm_v2.yaml")
my_model_path = os.path.join(model_root, "hfd_100epochs.ckpt")


device = "cuda:0"

# # Chinese-English dictionary
category_dict ={
    "连衣裙": "Dress",
    "衬衣": "Blouse",
    "毛衣": "Sweater",
    "外套": "Coat",
    "连身裤": "Jumpsuit",
    "裤子": "Pant",
    "裙子": "Skirt"
}

style_dict = {
    "原创设计": "Original Design",
    "街头": "Street Style",
    "通勤": "Commute",
    "韩版": "Korean Style",
    "淑女": "Elegant",
    "甜美": "Sweet",
    "欧美": "Western Style",
    "日系": "Japanese Style",
    "英伦": "British Style",
    "复古": "Vintage",
    "文艺": "Artsy",
    "宫廷": "Courtly Style",
    "简约": "Simple Style",
    "乡村": "Rural Style",
    "学院风": "Campus Style",
    "OL": "Office Lady",
    "居家": "Homewear",
    "运动": "Sport",
    "休闲": "Casual",
    "高贵": "Noble",
    "青春/流行": "Youth/Pop",
    "商务绅士": "Business Gentleman",
    "清新": "Fresh",
    "潮/潮流/新潮": "Trendy",
    "中国风": "Chinese Style",
    "朋克": "Punk",
    "嘻哈": "Hip-hop",
    "性感": "Sexy",
    "摇滚": "Rock",
    "工装": "Workwear",
    "职场": "Office"
}

occasion_dict = {
    "校园": "Campus",
    "居家": "Home",
    "约会": "Date",
    "聚会": "Party",
    "职场": "Office",
    "运动": "Sport",
    "旅行": "Travel",
    "婚礼": "Wedding",
    "商务": "Business"
}

effect_dict = {
    "修身显瘦": "Slimming",
    "显年轻": "Youthful",
    "显高": "Tall",
    "显臀": "Highlight Hips",
    "显白": "Brighten Skin",
    "显脸小": "Face Slimming",
    "显脖子长": "Elongate Neck",
    "显胸": "Enhance Bust",
    "显壮男": "Muscular Look"
}

feeling_dict = {
    "弧度感": "Sense of Curve",
    "灵动飘逸感": "Sense of Agility and Elegance",
    "束缚感": "Sense of Restraint",
    "立体层次感": "Three-dimensional Layering",
    "朦胧感": "Hazy Sensation",
    "垂坠感": "Drape Feeling",
    "沉闷感": "Dullness Sensation",
    "俏皮感": "Playful Feeling",
    "青春感": "Youthful Feeling",
    "趣味感": "Sense of Fun",
    "轻松随意感": "Casual and Relaxed Feeling",
    "大气感": "Sense of Atmosphere",
    "线条感": "Line Feeling",
    "堆叠感": "Stacking Feeling",
    "成熟感": "Mature Feeling",
    "童真感": "Childlike Feeling",
    "臃肿感": "Bulky Feeling",
    "挺括感": "Crisp Feeling",
    "负重感": "Heavy Feeling",
    "丛林感": "Jungle Feeling",
    "重工感": "Heavy Industry Feeling"
}


attribute_dict = {
    "衣长": "A1",
    "袖长": "A2",
    "袖型": "A3",
    "领型": "A4",
    "下摆": "A5"
}

clothing_length_dict = {
    "超短款": "Ultra-short",
    "短款": "Short",
    "中款": "Knee-length",
    "中长款": "Mid-length",
    "长款": "Long"
}

sleeve_length_dict = {
    "无袖": "Sleeveless",
    "短袖": "Short Sleeve",
    "中袖": "Elbow-length Sleeve",
    "中长袖": "Mid-length Sleeve",
    "长袖": "Long Sleeve"
}

sleeve_type_dict = {
    "蝙蝠袖": "Dolman Sleeve",
    "泡泡袖": "Puffed Sleeve",
    "灯笼袖": "Lantern Sleeve",
    "喇叭袖": "Flare Sleeve",
    "插肩袖": "Raglan Sleeve",
    "荷叶袖": "Ruffle Sleeve",
    "包袖": "Wrapped Sleeve",
    "牛角袖": "Raglan Sleeve",
    "飞飞袖": "Flutter Sleeve",
    "公主袖": "Princess Sleeve",
    "堆堆袖": "Layered Sleeve",
    "衬衫袖": "Shirt Sleeve",
    "花瓣袖": "Petal Sleeve",
    "连袖": "Kimono Sleeve",
    "常规袖": "Regular Sleeve",
    "落肩袖": "Drop-shoulder Sleeve",
}

collar_type_dict = {
    "圆领": "Round Collar",
    "V领": "V-Neck",
    "方领": "Square Collar",
    "驳领": "Tailor Collar",
    "翻领": "Lapel Collar",
    "立领": "Stand Collar",
    "T领": "T-neck",
    "一字领": "Boat Neck",
    "U领": "U-Neck",
    "A字领": "A-Line Collar",
    "荡领": "Swinging Collar",
    "不规则领": "Irregular Collar",
    "关门领": "Closed Collar"
}

hem_dict = {
    "平下摆": "Flat Hem",
    "圆弧下摆": "Curved Hem",
    "荷叶下摆": "Ruffle Hem",
    "层叠下摆": "Layered Hem",
    "低腰下摆": "Low Waist Hem",
    "条纹下摆": "Striped Hem",
    "波浪下摆": "Wavy Hem",
    "开衩下摆": "Slit Hem",
    "垂坠下摆": "Draped Hem",
    "不规则下摆": "Irregular Hem",
    "卷边下摆": "Curled Hem",
    "毛边下摆": "Raw Hem",
    "束脚下摆": "Ankle-tied Hem",
    "开衩下摆": "Slit Hem",
    "喇叭下摆": "Flared Hem",
    "翻边下摆": "Flanging Hem",
    "宽松下摆": "Loose Hem",
    "花边下摆": "Lace Hem",
    "收紧带下摆": "Tight-strap Hem",
    "带抽绳下摆": "Drawstring Hem",
    "百褶下摆": "Pleated Hem"
}