import requests
# x = requests.get('https://www.reddit.com/r/photoshopbattles/top/.json?count=20', headers = {'User-agent': 'photoshopbot'})
# print(x.status_code)
# x = x.json()
# print(x[1])

def getPosts(url, limit, ispost):
    top_Posts = requests.get(url+'/top/.json?count='+str(limit), headers = {'User-agent': 'photoshopbot'})
    if top_Posts.status_code != 200:
        print (top_Posts.status_code)
    else:
        top_Posts=top_Posts.json()
        posts_dict={}
        if ispost
        for post_number in range(0, limit):
            post_id=top_Posts['data']['children'][post_number]['data']['id']
            post_img=top_Posts['data']['children'][post_number]['data']['url']
            posts_dict[post_id]=post_img
            download = wget.download(post_img)
        print(posts_dict.keys())

if __name__ == "__main__":
    getPosts('https://www.reddit.com/r/photoshopbattles',10)    
