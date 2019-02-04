class DecisionTree:

	def __init__(self):
		self.dataset = 0
	
	def set_attribute(self, data):
		self.dataset = data

	def get_attribute(self):
		return self.dataset

	classData = property(get_attribute, set_attribute)


class Operations:

	def __init__(self,data):
		self.set = data

	""" SPLIT THE DATA INTO RIGHT AND LEFT"""

	def test_split(self, index, value, dataset):
		left, right = [], []
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)

		return left, right

	""" FIND THE BEST SPLIT VALUES FROM THE  DATASET"""

	def gini_index(self, groups, values):
		#count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		gini = 0.0
		for group in groups:
			size = float(len(group))

			if size == 0:
				continue

			score = 0.0

			for class_val in values:
				p = [row[-1] for row in group].count(class_val) / size
				score += p * p

			gini += (1.0 - score) * (size / n_instances)

		return gini

	""" FIND THE BEST SPLIT OF DATA"""

	def split_data(self):
		
		#separate the input features and label from the data
		class_values = list(set(data[-1] for data in self.set))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		
		for index in range(len(self.set[0]) - 1):
			for row in self.set:
				groups = self.test_split(index, row[index], self.set)
				gini = self.gini_index(groups, class_values)

				print("{} < {}, Gini = {}".format((index +  1), row[index], gini))

				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups


		return {'index': b_index, 'value': b_value, 'Score': b_score, 'Groups': b_groups}


	"""CREATE A TERMINAL NODE VALUE"""
	def to_terminal(self, group):

		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key= outcomes.count)

	"""CREATE CHILD SPLITS FOR A NODE  OR MAKE TERMINAL"""

	def split(self, node, max_depth, min_size, depth):
		left, right = node['Groups']
		#check for no split
		if not left or not right:
			node['left'] = node['right'] = self.to_terminal(left + right)
			return

		#check for max-depth
		if depth >= max_depth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)

		#process left child
		if len(left) <= min_size:
			node['left'] = to_terminal(left)
		else:
			node['left'] = self.split_data(left)
			self.split(node['left'], max_depth, min_size, depth + 1)


		#process right child
		if len(right) <= min_size:
			node['right'] = to_terminal(right)
		else:
			node['right'] = self.split_data(right)
			self.split(node['right'], max_depth, min_size, depth + 1)

	"""BUILDING A TREE"""
	def buildTree(self, max_depth, min_size):
		root = self.split_data()
		self.split(root, max_depth, min_size, 1)

		return root



#random dataset
dataset = [[4.5667, 5.667, 7.0000, 0],
	[3.5567, 2.9990, 6.0000, 0],
	[5.667, 3.0000, 2.4567, 0],
	[6.0098, 2.00986, 5.67854, 1],
	[7.9890, 8.9904, 6.89879, 1],
	[9.08745, 7.74783, 3.739843, 1],
	[5.87498, 8.78678, 10.6768787, 2],
	[4.87867, 7.787656, 9.987876, 2],
	[3.7767867, 8.89787, 10.676565, 2]]

dt = DecisionTree()
dt.classData = dataset #set the dataset to the object
op = Operations(dt.classData)
dictDump = op.buildTree(4, 1)
