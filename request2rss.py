import requests

def getPosts(url, limit):
    top_Posts = requests.get(url+'/top/.json?count='+str(limit), headers = {'User-agent': 'photoshopbot2'})
    if top_Posts.status_code != 200:
        print (top_Posts.status_code)
    else:
        top_Posts=top_Posts.json()
        posts_dict={}
        for post_number in range(0, limit):
            post_id=top_Posts['data']['children'][post_number]['data']['id']
            post_img=top_Posts['data']['children'][post_number]['data']['url']
            posts_dict[post_id]=post_img
            #print(post_img.rsplit("/",1)[1])
            image = requests.get(post_img, allow_redirects=True)
            open('data/postImage/'+post_img.rsplit("/",1)[1], 'wb').write(image.content)
    return posts_dict

if __name__ == "__main__":
    getPosts('https://www.reddit.com/r/photoshopbattles',10)    
