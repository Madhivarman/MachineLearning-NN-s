import string

def load_dataset(filename):
	doc = open(filename).read()
	mapping = dict()

	for line in doc.split("\n"):
		tokens = line.split() #split by white space
		#skip the empty line
		if len(line) < 1:
			continue
		
		image_name,image_desc = tokens[0],tokens[1:]
		image_name = image_name.split('.')[0] #remove jpg file
		image_desc = ' '.join(image_desc)

		if image_name not in mapping:
			mapping[image_name] = list() #create image_name as a key and add description as a values

		mapping[image_name].append(image_desc)

	return mapping

def clean_description(desc):
	table = str.maketrans('','',string.punctuation) #prepare translation table for punctuation
	for key,desc_list in desc.items():
		for i in range(len(desc_list)):
			descr = desc_list[i]
			descr = descr.split() #tokenize
			descr = [word.lower() for word in descr] #convert into lower
			descr = [w.translate(table) for w in descr] #remove punctuation
			descr = [word for word in descr if len(word) > 1] #remove hanging words
			descr = [word for word in descr if word.isalpha()] #remove tokens with number

			desc_list[i] = ' '.join(descr)

def to_vocabulary(descriptions):
	all_desc = set() #build all set of words
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]

	return all_desc

def save_descriptions(descriptions,filename):
	lines = list()
	for key,desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key+' '+desc)
	data = '\n'.join(lines)
	file = open(filename,'w')
	file.write(data)

	file.close() #close the data	

def main():
	#load the dataset
	train_dataset = 'Flickr8k_text/Flickr8k.token.txt'
	train = load_dataset(train_dataset)
	print("Length of the Train Images:{}".format(len(train)))
	#parse description
	clean_description(train)
	#summarize to vocabulary
	vocabulary = to_vocabulary(train)
	print("vocabulary size:{}".format(len(vocabulary)))

	save_descriptions(train,'descriptions.txt')
	print("File is Written..!")


if __name__ == '__main__':
	main()