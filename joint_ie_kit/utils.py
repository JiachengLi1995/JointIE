

class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.id2label = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.id2label[self.label_num] = label
        self.label_num += 1

        return self.label2id[label]

    def get_label(self, id):

        if id not in self.id2label:
            print(f'Cannot find label that id is {id}!!!')
            assert 0
        return self.id2label[id]
    
    def get_num(self):
        return self.label_num
    
    def all_labels(self):
        return list(self.label2id.keys())


