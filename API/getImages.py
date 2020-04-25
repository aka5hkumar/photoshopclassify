# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:32:28 2020

@author: njord
"""
#!/usr/bin/env python
# coding: utf-8
import requests
import os
import imgur_downloader as Imgur

def getImages(url, search_filter, limit, posts_dict=None):
    if not posts_dict: # If no posts are passed, get post IDs and Images
        print('Getting Posts')
        resp = requests.get(url+'/'+search_filter+'/.json?count='+str(limit), headers = {'User-agent': 'photoshopbot2'})
        if resp.ok:
            resp_json=resp.json()
            posts_dict={}
            for post_number in range(0, limit):
                post_id=resp_json['data']['children'][post_number]['data']['id']# get ID
                post_img=resp_json['data']['children'][post_number]['data']['url']# get img link
                posts_dict[post_id]=post_img # add to Dict                
                image = requests.get(post_img, allow_redirects=True, stream=True) #Request image
                filename = ('./data/original'+'/p_'+post_img.rsplit("/",1)[1])
                #filename = ('./data/Images/'+post_id+'/p_'+post_img.rsplit("/",1)[1])
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                open('./data/original'+'/p_'+post_img.rsplit("/",1)[1], 'wb').write(image.content)
                #open('./data/Images/'+post_id+'/p_'+post_img.rsplit("/",1)[1], 'wb').write(image.content) #save image
            getImages(url,search_filter,limit,posts_dict) # Recursive call with post dicitonary
        else:
            print (resp.reason)
    else: # If posts are passed, get comment IDs and Images for each post
        print('Getting Comments')
        for post_id in posts_dict.keys():
            resp = requests.get(url+'/comments/'+post_id+'.json?depth=1', headers = {'User-agent': 'photoshopbot2'})
            if resp.ok:
                resp_json=resp.json()
                comment_dict={}
                for count, commment in enumerate(resp_json[1]['data']['children']):
                    if count == 9: # count starts at 0
                        break
                    else:    
                        comm_id=commment['data']['id']
                        try:
                            comm_img=commment['data']['body_html'].rsplit('"')[3]
                            if not(comm_img.endswith(('.jpg','.png','.gif', 'jpeg'))):
                                if comm_img[-1] == '/':                                    pass
                                else:
                                    filename = ('./data/photoShopped/'+post_id+comm_img.rsplit("/",1)[1])
                                    #filename = ('./data/Images/'+post_id+'/commentImage/'+comm_img.rsplit("/",1)[1])
                                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                                    try:
                                        Imgur.ImgurDownloader(comm_img, './data/photoShopped').save_images()
                                        #Imgur.ImgurDownloader(comm_img, './data/Images/'+post_id+'/commentImage').save_images()
                                    except:
                                        count -= 1
                                        pass
                                #print(comm_img.rsplit('rel="image_src" href=')[1])
                            else:          
                                image = requests.get(comm_img, allow_redirects=True, stream=True) #Request image
                                filename = ('./data/photoShopped/'+post_id+comm_img.rsplit("/",1)[1])
                                #filename = ('./data/Images/'+post_id+'/commentImage/'+comm_img.rsplit("/",1)[1])
                                os.makedirs(os.path.dirname(filename), exist_ok=True)
                                open('./data/photoShopped/'+post_id+comm_img.rsplit("/",1)[1], 'wb').write(image.content)
                                #open('./data/Images/'+post_id+'/commentImage/'+comm_img.rsplit("/",1)[1], 'wb').write(image.content) #save image
                        except IndexError:
                            count -= 1
                            pass
                    
            else:
                print (resp.reason)
if __name__=='__main__':     
    url = 'https://www.reddit.com/r/photoshopbattles'
    limit = 10
    search_filter = 'top'
    isPost=True
    
    getImages(url,search_filter,limit)


