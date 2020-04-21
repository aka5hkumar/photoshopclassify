# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:39:52 2020

@author: njord
"""

#!/usr/bin/env python
# coding: utf-8
import requests
import json
import os

class redditGetImages():
    def __init__(self,url,search_filter,limit):
        self.url = url 
        self.limit = limit
        self.search_filter = search_filter
        
    def getPosts(self): # Get post IDs and respective images, returns one list
        resp_post = requests.get('{}/{}.json?t=all&limit={}'.format(self.url,self.search_filter,self.limit)) 
        posts_list=[]
        if str(resp_post)=='<Response [200]>':
            resp_json=json.loads(resp_post.text)
            for post_number in range(0,self.limit):
                post_id=resp_json['data']['children'][post_number]['data']['id']
                post_img=resp_json['data']['children'][post_number]['data']['url']
                posts_list.append([post_id,post_img])
            return posts_list
        else:
            print(resp_post)
            return -1

    def getComments(self,post_id): # Get post IDs and respective images, returns one list
        resp_comm = requests.get('{}/comments/{}.json?depth=1'.format(self.url,post_id))
        comms_list=[]
        if str(resp_comm)=='<Response [200]>':
            resp_comm_json=json.loads(resp_comm.text)
            for comment_number in range(0,10):
                comm_img=resp_comm_json[1]['data']['children'][comment_number]['data']['body']
                comm_img=self.cleanComment(comm_img)
                comm_id=resp_comm_json[1]['data']['children'][comment_number]['data']['id']
                comms_list.append([post_id,comm_id,comm_img])
            return comms_list
        else:
            print('{} getting comments for post ID: {}'.format(resp_comm,post_id))
            return -1
            
    def cleanComment(self,comment):# Clean comment for URL. This needs working on 
        if comment.find("(")==-1: 
            return comment
        else:
            comment=comment[comment.find("(")+1:comment.find(")")]
            return comment
        
    
    def downloadPostImages(self,posts):
        for post in posts:
            url = post[1]
            post_id = post[0]
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    with open(os.getcwd()+'/images/posts/'+post_id+'.jpg', 'wb') as file:
                        file.write(resp.content)
                        print('Saved Image to '+'/images/posts/'+post_id+'.jpg')
            except:
                print('Failed to download '+url)
    
    def downloadCommmentImages(self,comments):
        for comment in comments:
            url = comment[2]
            comment_id = comment[1]
            post_id = comment[0]
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    with open(os.getcwd()+'/images/comments/'+post_id+'_'+comment_id+'.jpg', 'wb') as file:
                        file.write(resp.content)
                        print('Saved Image to '+'/images/comments/'+post_id+'_'+comment_id+'.jpg')
            except:
                print('Failed to download '+url)

    def downloadAll(self):
        post_list = self.getPosts()
        
        # Loop through comments of posts. I expect API to return 429 because of how much we are hitting it
        batch_comments=[]
        for post in post_list:
            post_comments = API.getComments(post[0])
            if post_comments != -1:
                batch_comments+=post_comments
        print(batch_comments)
        # Save post images
        self.downloadPostImages(post_list)
        
        # Save Comment Images
        self.downloadCommmentImages(batch_comments)
        
        
        
        
url = 'https://www.reddit.com/r/photoshopbattles'
limit = 10
search_filter = 'top'

API = redditGetImages(url,search_filter,limit)
API.downloadAll()



