import os

import os

test_list = [line. rstrip('\n') for line in open('./food-101/meta/test.txt')]
os.mkdir('./food-101/test')
source_base = './food-101/images/'
target_base = './food-101/test/'
for item in test_list:
    c = item.split('/')[0]
    if not os.path.exists(os.path.join(base, c)):
        os.mkdir(os.path.join(base, c))
    os.rename(os.path.join(source_base, item) + '.jpg', os.path.join(target_base, item) + '.jpg')