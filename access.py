import mysql.connector 
import json 
import pandas as pd

mydb = mysql.connector.connect(
	host="localhost",
	user="boyko",
	passwd="boykoboyko1",
	database="labeller"
)

# {"label_id":"6","image_id":"1","username":"fa344","type":"parasite","x1":"2083","x2":"2197","y1":"839","y2":"957","updated":"2019-01-23 18:59:17"},

def read_image_db():
	imgcursor = mydb.cursor()
	imgcursor.execute("SELECT * FROM images")
	for x in imgcursor:
		print(x)

def read_json():
	url = "https://filip.ayazi.sk/labeller/api.php?key=b7ca50e53329c&data&fbclid=IwAR3UNpOwXw3BaePHLMrsrKhKONRbx8AdbH7GOmYhEjA1P42jYqV3k44OUF0"
	df = pd.read_json(url)
	print(df)

def to_json(cursor):
	None

def get_entries_fromtable(tablename):
	sqlln = "SELECT * FROM {}".format(tablename)
	cursor = mydb.cursor()
	cursor.execute(sqlln)
	return cursor

# {"image_id":"1","url":"images\/5c48b94bc13395c48b94bc13a35c48b94bc140f.jpg","type":"boyko1","filename":"image_001.jpg","process":"1000","added":"2019-01-23 18:58:19","priority":"50","labels":
	
def get_images_df():
	out_cursor = get_entries_fromtable("images")
	df_dict = {
		"image_id":[],
		"url":[],
		"type":[],
		"filename":[],
		"process":[],
		"added":[],
		"priority":[],
		"labels":[] # list of json entries
	}
	for image in out_cursor:
		df_dict["image_id"].append(image[0])
		df_dict["url"].append(image[1])
		df_dict["type"].append(image[2])
		df_dict["filename"].append(image[3])
		df_dict["process"].append(image[4])
		df_dict["added"].append(image[5].strftime("%Y-%m-%d %H:%M:%S"))
		df_dict["priority"].append(image[6])
		df_dict["labels"].append([])

	return pd.DataFrame.from_dict(df_dict)

def label_to_json_string(label_tuple):
	label_dict = {
		"label_id": label_tuple[0],
		"image_id": label_tuple[1],
		"username": label_tuple[2],
		"type": label_tuple[3],
		"x1": label_tuple[4],
		"x2": label_tuple[5],
		"y1": label_tuple[6],
		"y2": label_tuple[7],
		"updated": label_tuple[8].strftime("%Y-%m-%d %H:%M:%S")
	}

	return json.dumps(label_dict)

def get_labels_df():
	ids_cursor = mydb.cursor()
	ids_cursor.execute("SELECT DISTINCT image_id FROM labels")
	ids = []
	for i in ids_cursor:
		ids.append(i[0])

	label_cursor = mydb.cursor()

	df_labels = {
		"image_id": [],
		"labels": [] # list of json strings
	}

	for index, value in enumerate(ids):
		df_labels["labels"].append([])
		df_labels["image_id"].append(value)
		label_cursor.execute("SELECT * FROM labels WHERE image_id='{}'".format(value))
		for label in label_cursor:
			df_labels["labels"][index].append(label_to_json_string(label))

	return pd.DataFrame.from_dict(df_labels)

def get_images_with_labels_df():
	images_df = get_images_df()
	labels_df = get_labels_df()

	for image_id in labels_df["image_id"]:
		labels = labels_df.loc[labels_df["image_id"] == image_id]["labels"]
		images_df.loc[images_df["image_id"] == image_id, "labels"] = labels

	return images_df
#read_json()
#read_image_db()
#get_images_with_labels_df()

if __name__ == "__main__":
	main()
