import requests
import os
import imgur_downloader as Imgur
import csv

class GetImages:
    def __init__(self, subreddit, post_limit, search_filter, comment_limit=float('inf')):
        self.url = 'https://www.reddit.com/r/'+subreddit
        self.post_limit = post_limit
        self.posts = post_limit
        self.comment_limit = comment_limit
        self.filter = search_filter

    def getPosts(self, after='', posts_dict={}, csv_array=[]):
        if self.post_limit<=0:
            self.getComments(posts_dict, csv_array) 
        else:
            resp = requests.get(self.url+'/'+self.filter+'/.json?sort='+self.filter+'&t=all&after='+after+'/count=20', headers = {'User-agent': 'photoshopbot2'})
            if resp.ok:
                print('Getting posts')
                resp_json=resp.json()
                self.post_limit = self.post_limit-len(resp_json['data']['children'])
                after = resp_json['data']['after']
                for count, _ in enumerate(resp_json['data']['children']):
                    post_id=resp_json['data']['children'][count]['data']['id']# get ID
                    post_img=resp_json['data']['children'][count]['data']['url']# get img link
                    posts_dict[post_id]=post_img # add to Dict                
                    image = requests.get(post_img, allow_redirects=True, stream=True) #Request image
                    filepath = ('./data/images/')
                    filename = ('o_'+post_img.rsplit("/",1)[1])
                    os.makedirs(os.path.dirname(filepath+filename), exist_ok=True)
                    open(filepath+filename, 'wb').write(image.content)
                    csv_array.append([filename.split('.')[0], 0])
                if self.post_limit > 0:
                    print(self.post_limit, 'posts remaining')
                self.getPosts(after, posts_dict, csv_array)
            else:
                print (resp.reason)

    def getComments(self, posts_dict, csv_array):
        for post_count, post_id in enumerate(posts_dict.keys()):
            resp = requests.get(self.url+'/comments/'+post_id+'.json?depth=1', headers = {'User-agent': 'photoshopbot2'})
            if resp.ok:
                resp_json=resp.json()
                percentComplete = str(int(100 * post_count/self.posts))
                print('Getting comments',percentComplete, '%')
                for count, commment in enumerate(resp_json[1]['data']['children']):
                    if count >= self.comment_limit: # count starts at 0
                        break
                    else:    
                        try:
                            comm_img=commment['data']['body_html'].rsplit('"')[3]
                            if not(comm_img.endswith(('.jpg','.png','.gif', 'jpeg'))):
                                if comm_img[-1] == '/':                                    
                                    pass
                                else:
                                    filepath = ('./data/images/')
                                    filename = ('p_'+post_id+comm_img.rsplit("/",1)[1])

                                    os.makedirs(os.path.dirname(filepath+filename), exist_ok=True)
                                    try:
                                        Imgur.ImgurDownloader(comm_img, filepath, filename).save_images()
                                        csv_array.append([filename, 1])
                                    except:
                                        count -= 1
                                        pass
                            else:          
                                image = requests.get(comm_img, allow_redirects=True, stream=True) #Request image
                                filepath = ('./data/images/')
                                filename = ('p_'+post_id+comm_img.rsplit("/",1)[1])
                                os.makedirs(os.path.dirname(filepath+filename), exist_ok=True)
                                open(filepath+filename, 'wb').write(image.content)
                                csv_array.append([filename.split('.')[0], 1])
                        except (IndexError, KeyError):
                            count -= 1
                            pass       
            else:
                print (resp.reason)
        #self.makeCSV(csv_array)
        return csv_array

def makeCSV(csv_array):
    with open("./data/data_labels.csv","w+",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(csv_array)
        print ('Writing CSV')
        #Does not remove duplicates!


def cleanImages():
    csv_list=[]
    for image in os.listdir('./data/images/'):
        if not (image.endswith(('jpg', 'png', 'jpeg'))):
            os.remove('./data/images/'+image)
        elif os.path.getsize('./data/images/'+image) < 20 * 1024:
            os.remove('./data/images/'+image)
        else:
            if image[0]=='o':
                csv_list.append([image,0])
            if image[0]=='p':
                csv_list.append([image,1])
    makeCSV(csv_list)


def main():
    subreddit = 'photoshopbattles'
    limit = 30
    search_filter = 'top'
    comment_limit = 5
    reddit = GetImages(subreddit, limit, search_filter,comment_limit)
    reddit.getPosts()
    cleanImages()
        

if __name__ == "__main__":
    main()