# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:32:28 2020

@author: njord
"""
#!/usr/bin/env python
# coding: utf-8
import requests
        
def getImages(url, search_filter, limit, posts_dict=None):
    if not posts_dict: # If no posts are passed, get post IDs and Images
        print('Getting Posts')
        resp = requests.get(url+'/'+search_filter+'/.json?count='+str(limit), headers = {'User-agent': 'photoshopbot2'})
        if resp.status_code == 200:
            resp_json=resp.json()
            posts_dict={}
            for post_number in range(0, limit):
                post_id=resp_json['data']['children'][post_number]['data']['id']# get ID
                post_img=resp_json['data']['children'][post_number]['data']['url']# get img link
                posts_dict[post_id]=post_img # add to Dict                
                image_capture = requests.get(post_img, stream=True) # Save image
                with open('./images/original/'+post_id+'.png', 'wb') as out_file:
                    out_file.write(image_capture.content)
            getImages(url,search_filter,limit,posts_dict) # Recursive call with post dicitonary
        else:
            print (resp.status_code)
            return 1
    else: # If posts are passed, get comment IDs and Images for each post
        print('Getting Comments')
        for post_id in posts_dict.keys():
            resp = requests.get(url+'/comments/'+post_id+'.json?depth=1', headers = {'User-agent': 'photoshopbot2'})
            if resp.status_code == 200:
                resp_json=resp.json()
                comment_dict={}
                for comment_number in range(0,10):
                    comm_id=resp_json[1]['data']['children'][comment_number]['data']['id']# get ID
                    comm_img=resp_json[1]['data']['children'][comment_number]['data']['body']# get img
                    if comm_img.find("(")==-1: # clean Comment
                        comm_img=comm_img[comm_img.find("(")+1:comm_img.find(")")]
                    else:
                        comm_img=comm_img[comm_img.find("(")+1:comm_img.find(")")]
                    comment_dict[comm_id]=comm_img # add to Dict
                    image_capture = requests.get(comm_img, stream=True) # Save 
                    with open('./images/photoshopped/'+post_id+comm_id+'.png', 'wb') as out_file:
                        out_file.write(image_capture.content)
            else:
                print (resp.status_code)
   
if __name__=='__main__':     
    url = 'https://www.reddit.com/r/photoshopbattles'
    limit = 10
    search_filter = 'top'
    isPost=True
    
    getImages(url,search_filter,limit)
