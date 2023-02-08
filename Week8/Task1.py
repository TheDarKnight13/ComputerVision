#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares


# In[2]:


#function to calculate the projective homography between two images
def proj_homography(dcord,rcord):
    A = np.zeros((8,8))
    B = np.zeros((8,1))
    H = np.ones((3,3))
 
    #Generating the matrix equation AB = C using the coordinates of the ROIs 
    for i in range(4):
        A[2*i,0] = dcord[i,0]
        A[2*i,1] = dcord[i,1]
        A[2*i,2] = 1
        A[2*i,6] = -1*dcord[i,0]*rcord[i,0]
        A[2*i,7] = -1*dcord[i,1]*rcord[i,0]
 
        A[2*i+1,3] = dcord[i,0]
        A[2*i+1,4] = dcord[i,1]
        A[2*i+1,5] = 1
        A[2*i+1,6] = -1*dcord[i,0]*rcord[i,1]
        A[2*i+1,7] = -1*dcord[i,1]*rcord[i,1]
 
        B[2*i,0] = rcord[i,0]
        B[2*i+1,0] = rcord[i,1]
 
 
    C = np.dot(np.linalg.pinv(A),B);
 
    #Obtaining the Homography H
    for i in range(3):
        for j in range(3):
            if i==2 & j ==2:
                break
            H[i,j]=C[3*i +j,0]
 
    return H


# In[3]:


#function to calculate the homography between two images using inhomogeneous linear least square
def LLS_homography(dcord,rcord):
    n = dcord.shape[0]
    A = np.zeros((2*n,8))
    B = np.zeros((2*n,1))
    H = np.ones((3,3))
 
    #Generating the matrix equation AB = C using the coordinates of the ROIs 
    for i in range(n):
        A[2*i,0] = dcord[i,0]
        A[2*i,1] = dcord[i,1]
        A[2*i,2] = 1
        A[2*i,6] = -1*dcord[i,0]*rcord[i,0]
        A[2*i,7] = -1*dcord[i,1]*rcord[i,0]
 
        A[2*i+1,3] = dcord[i,0]
        A[2*i+1,4] = dcord[i,1]
        A[2*i+1,5] = 1
        A[2*i+1,6] = -1*dcord[i,0]*rcord[i,1]
        A[2*i+1,7] = -1*dcord[i,1]*rcord[i,1]
 
        B[2*i,0] = rcord[i,0]
        B[2*i+1,0] = rcord[i,1]
 
 
    C = np.dot(np.matmul(np.linalg.pinv(A.T@A),A.T),B);
 
    #Obtaining the Homography H
    for i in range(3):
        for j in range(3):
            if i==2 & j ==2:
                break
            H[i,j]=C[3*i +j,0]
 
    return H


# In[4]:


#Returns the inliers for a given homography H
def theinliers(H,im1,im2,delta):
    him1 = np.insert(im1,2,1,axis=1)
    him2 = np.insert(im2,2,1,axis=1)
    
    him2_calc = np.matmul(H,him1.T)
    im2_calc = him2_calc/him2_calc[2,:]
    im2_calc = im2_calc[0:2,:].T
    error  = np.abs(im2-im2_calc)
    t_error = np.sum(error**2,axis=1)
    
    idx = np.where(t_error<delta)[0]
    
    return idx
    


# In[5]:


#Implementation of RANSAC and resultant homography is refined using inbulit LM method
def RANSAC(im1,im2):
    delta = 3.0
    p = 0.99
    epsilon = 0.75
    n = 4
    ntotal = im1.shape[0]
    N = np.log(1-p)/np.log(1-(1-epsilon)**n)
    N = int(N)
    M = (1-epsilon)*ntotal    
  
    most_inliers = []
    max_inliers=0
    
    for i in range(N):
        idx = np.random.choice(list(range(ntotal)),n)
        
        Dcord=[]
        Rcord=[]
        for j in idx:
            Dcord.append(im1[j])
            Rcord.append(im2[j])
        dcord = np.asarray(Dcord)
        rcord = np.asarray(Rcord)
        H = proj_homography(dcord,rcord)
        inliers = theinliers(H,im1,im2,delta**2)
        if len(inliers)>M and len(inliers)>max_inliers:
            max_inliers = len(inliers)
            most_inliers = inliers
            
    H_final = LLS_homography(im1[most_inliers],im2[most_inliers])
    H_finals = LM_inbuilt(im1[most_inliers],im2[most_inliers],H_final)
        
    return H_finals
               

    
    


# In[6]:


#Cost function of LM method
def cost_LM(h,dcoord,rcoord):
    H = h.reshape((3,3))
    hdcoord = np.insert(dcoord,2,1,axis=1)
    hrcoord = np.insert(rcoord,2,1,axis=1)
    
    hrcoord_calc = np.matmul(H,hdcoord.T)
    hrcoord_calc = hrcoord_calc/hrcoord_calc[2,:]
    hrcoord_calc = hrcoord_calc.T
    cost  = np.abs(hrcoord-hrcoord_calc)
    
    return cost.sum(axis=1)**2
    


# In[7]:


#Inbuilt LM function
def LM_inbuilt(dcoord,rcoord,H):
    h  = H.flatten()
    LM = least_squares(cost_LM,h,args=(dcoord,rcoord),method="lm")
    h_LM = LM.x
    H_LM = h_LM.reshape((3,3))
    return H_LM
    


# In[8]:


#To select the relevant lines from all the Hough lines
def selected_lines(lines, x11,x12,x21,x22, ver):
    if ver:
        dist = lines[:,0]*np.cos(lines[:,1])
        thresh = 0.05*(np.max(np.abs(dist))-np.min(np.abs(dist)))
    else:
        dist = lines[:,0]*np.sin(lines[:,1])
        thresh = 0.05*(np.max(np.abs(dist))-np.min(np.abs(dist)))
        
    sort = np.argsort(dist,axis =0)
    dist_sort = dist[sort]
    prim = []
    sec = []
    for i in range(dist_sort.shape[0]) :
        if i == 0:
            sec.append(sort[i])
        elif dist_sort[i]-dist_sort[i-1] < thresh :
            sec.append(sort[i])
        else :
            prim.append(sec)
            sec = [sort[i]]
        
    prim.append(sec)
    
    x11_list = []
    x12_list = []
    x21_list = []
    x22_list = []
    for i in prim:
        x11_list.append(np.mean(x11[i],axis=0).astype('int32'))
        x12_list.append(np.mean(x12[i],axis=0).astype('int32'))
        x21_list.append(np.mean(x21[i],axis=0).astype('int32'))
        x22_list.append(np.mean(x22[i],axis=0).astype('int32'))
        
    return np.array(x11_list), np.array(x12_list), np.array(x21_list),np.array(x22_list)

        


# In[9]:


#Input of the image
def inter_pts(img,plot=False):
    if(plot):
        f = plt.figure(figsize=(15,9))
        
        ax1 = f.add_subplot(221)
        ax2 = f.add_subplot(222)
        ax3 = f.add_subplot(223)
        ax4 = f.add_subplot(224)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        ax1.title.set_text('Edge Detection')
        ax2.title.set_text('All Hough Lines')
        ax3.title.set_text('Selected Hough lines')
        ax4.title.set_text('Corners')
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gimg = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(3,3),1.4)

    fimg = cimg.copy()
    fimg1 = cimg.copy()
    can = cv2.Canny(gimg,255*1.5 ,255) #Canny output
    if(plot):
        ax1.imshow(can,cmap='gray')
        

    #Plotting the horizontal and vertical Hough lines
    hlines = cv2.HoughLines(can, 1, np.pi / 180, 50, None, 0, 0)
    lines = hlines.reshape(hlines.shape[0],hlines.shape[2])
    r = lines[:,0]
    theta = lines[:,1]
    a = np.cos(theta)
    b = np.sin(theta)
    x11 = (a*r-1000*b).astype('int32')
    x12 = (b*r+1000*a).astype('int32')
    x21 = (a*r+1000*b).astype('int32') 
    x22 = (b*r-1000*a).astype('int32')
    
    for i in range(len(x11)):
        cv2.line(fimg1,(x11[i],x12[i]),(x21[i],x22[i]),(0,0,0),1)
        
    if(plot):
        ax2.imshow(fimg1)
        

    ovx11 = x11[np.where(b**2<0.5)]
    ovx12 = x12[np.where(b**2<0.5)]
    ovx21 = x21[np.where(b**2<0.5)]
    ovx22 = x22[np.where(b**2<0.5)]

    vx11,vx12,vx21,vx22 = selected_lines(lines[np.where(b**2<0.5)],ovx11,ovx12,ovx21,ovx22,True)

    ohx11 = x11[np.where(b**2>0.5)]
    ohx12 = x12[np.where(b**2>0.5)]
    ohx21 = x21[np.where(b**2>0.5)]
    ohx22 = x22[np.where(b**2>0.5)]

    hx11,hx12,hx21,hx22 = selected_lines(lines[np.where(b**2>0.5)],ohx11,ohx12,ohx21,ohx22,False)


    for i in range(len(vx11)):
        cv2.line(fimg,(vx11[i],vx12[i]),(vx21[i],vx22[i]),(0,255,0),1)
    
    for i in range(len(hx11)):
        cv2.line(fimg,(hx11[i],hx12[i]),(hx21[i],hx22[i]),(0,0,255),1)
    
    if(plot):
        ax3.imshow(fimg) #All Hough lines

    #Finding the intersections
    hp1 = np.append([hx11],[hx12],axis=0)
    hp1 = np.append(hp1,np.ones((1,hp1.shape[1])),axis=0).T

    hp2 = np.append([hx21],[hx22],axis=0)
    hp2 = np.append(hp2,np.ones((1,hp2.shape[1])),axis=0).T

    vp1 = np.append([vx11],[vx12],axis=0)
    vp1 = np.append(vp1,np.ones((1,vp1.shape[1])),axis=0).T

    vp2 = np.append([vx21],[vx22],axis=0)
    vp2 = np.append(vp2,np.ones((1,vp2.shape[1])),axis=0).T

    vl = np.cross(vp1,vp2)
    hl = np.cross(hp1,hp2)

    points=[]

    for i in range(vl.shape[0]):
        point = np.cross(hl,vl[i])
        points.append((point[:,:2]/point[:,2].reshape(-1,1)))

    pts = np.concatenate(points,axis=0)
    
    pimg = cimg.copy()
    for i in range (pts.shape [0]):
        if np.abs(pts[i,0].item())==float('inf'):
            pts = np.ones((10,2))
            break
        point = (int(pts[i,0].item()),int(pts[i,1].item()))
        pimg = cv2.circle(pimg,point,radius =3,color=(255,0,0),thickness = -1)
        pimg = cv2.putText(pimg,str(i),point,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
        
    if(plot):
        ax4.imshow(pimg)
    pts[:,[0,1]] = pts[:,[1,0]]
    
    return pts


# In[10]:


#Wrapper function for the homography
def Homography_wrapper(img,plot):
    pts = inter_pts(img,plot)
    if(pts.shape[0]!=80):
        return None,None
    else:
        xx,yy = np.meshgrid(np.arange(0,10),np.arange(0,8))
        rcoords = np.vstack((xx.reshape(-1),yy.reshape(-1))).T
        H = RANSAC(rcoords*10,pts)
        return H,pts


# In[11]:


#Calculates the image of the absolute conic
def Calc_omega(H_all):
    n = len(H_all)
    V = np.zeros((2*n,6))   
    
 
    #Generating the matrix equation Vb = 0 
    for i in range(n):
        V[2*i,0] = H_all[i][0][0]*H_all[i][0][1]
        V[2*i,1] = H_all[i][0][0]*H_all[i][1][1] + H_all[i][1][0]*H_all[i][0][1]
        V[2*i,2] = H_all[i][1][0]*H_all[i][1][1]
        V[2*i,3] = H_all[i][2][0]*H_all[i][0][1] + H_all[i][0][0]*H_all[i][2][1]
        V[2*i,4] = H_all[i][2][0]*H_all[i][1][1] + H_all[i][1][0]*H_all[i][2][1]
        V[2*i,5] = H_all[i][2][0]*H_all[i][2][1]
 
        V[2*i+1,0] = H_all[i][0][0]*H_all[i][0][0] - H_all[i][0][1]*H_all[i][0][1]
        V[2*i+1,1] = 2*H_all[i][0][0]*H_all[i][1][0]  -2*(H_all[i][0][1]*H_all[i][1][1])
        V[2*i+1,2] = H_all[i][1][0]*H_all[i][1][0] - H_all[i][1][1]*H_all[i][1][1]
        V[2*i+1,3] = 2*H_all[i][2][0]*H_all[i][0][0] - 2*(H_all[i][2][1]*H_all[i][0][1])
        V[2*i+1,4] = 2*H_all[i][2][0]*H_all[i][1][0] -2*(H_all[i][2][1]*H_all[i][1][1])
        V[2*i+1,5] = H_all[i][2][0]*H_all[i][2][0] - H_all[i][2][1]*H_all[i][2][1]

     
    u,s,vh = np.linalg.svd(V)
    b = vh[-1]
    
    omega = np.array([[ b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]]])
    
    return omega
 
    
    


# In[12]:


#Calculates the intrinsic Camera parameters
def Calc_K(H_list):
    o = Calc_omega(H_list)
    K = np.zeros((3,3))
    K[1,2] = (o[0,1]*o[0,2]-o[0,0]*o[1,2])/(o[0,0]*o[1,1]-o[0,1]*o[0,1])
    lamda = o[2,2] -((o[0,2]*o[0,2]+K[1,2]*(o[0,1]*o[0,2]-o[0,0]*o[1,2]))/(o[0,0]))
    K[0,0] = np.sqrt(lamda/o[0,0])
    K[1,1] = np.sqrt((lamda*o[0,0])/(o[0,0]*o[1,1]-o[0,1]*o[0,1]))
    K[0,1] = -(o[0,1]*K[0,0]*K[0,0]*K[1,1])/lamda
    K[0,2] = (K[0,1]*K[1,2]/K[1,1]) - (o[0,2]*K[0,0]*K[0,0]/lamda)
    K[2,2] = 1.0
    
    return K


# In[13]:


#Calculates the extrinsic Camera parameters
def Calc_Rt(H_nonull,K):
    H_list = np.array(H_nonull)
    h1 = H_list[:,:,0].T
    h2 = H_list[:,:,1].T
    h3 = H_list[:,:,2].T
    Kinv = np.linalg.pinv(K)
    r1t = Kinv@h1
    r2t = Kinv@h2
    tt = Kinv@h3
    eps = 1/(np.sqrt((r1t*r1t).sum(axis=0)))
    r1 = eps*r1t
    r2 = eps*r2t
    r3 = np.cross(r1.T,r2.T).T
    t = eps*tt
    u,d,v = np.linalg.svd(np.array([r1,r2,r3]).T)
    return t.T, np.matmul(u,v)
    
    
    


# In[14]:


#Calculating reprojection error
def mean_var_error(ptso,pts1):
    cost = np.abs(ptso-pts1)
    error = cost.sum(axis=1)**2
    return error.mean(),np.var(error)
    


# In[15]:


#Plots the original and the reprojected points
def Plot_points(H_fixed,i1,K,R,t,K_new,R_new,t_new,l):
    Rt = np.concatenate([[R[:,:,0]],[R[:,:,1]],[t.reshape((R.shape[0],-1))]],axis=0)
    RT = np.einsum('ijk->jki', Rt)
    H_recon = np.matmul(K,RT)
    H_final_recon = (H_recon.T/H_recon.max(axis=1).max(axis=1)).T
    
    Rt_new = np.concatenate([[R_new[:,:,0]],[R_new[:,:,1]],[t_new.reshape((R_new.shape[0],-1))]],axis=0)
    RT_new = np.einsum('ijk->jki', Rt_new)
    H_recon_new = np.matmul(K_new,RT_new)
    H_final_recon_new = (H_recon_new.T/H_recon_new.max(axis=1).max(axis=1)).T
    
    imgo = cv2.imread("HW8-Files/Dataset1/Pic_"+str(5)+".jpg")
    ptso = inter_pts(imgo,False)
    ptso[:,[0,1]] = ptso[:,[1,0]]
    pimg = cv2.cvtColor(imgo,cv2.COLOR_BGR2RGB)
    
    
    H = H_final_recon[i1]@np.linalg.pinv(H_fixed)
    H_new = H_final_recon_new[i1]@np.linalg.pinv(H_fixed)
    
    img1 = cv2.imread("HW8-Files/Dataset1/Pic_"+str(l[i1]+1)+".jpg")
    pt1 = inter_pts(img1,False)    
    ones = np.ones((pt1.shape[0],1))
    hdcord = np.append(pt1,ones,1).T
    
    
    temp = np.linalg.pinv(H/H.max())@hdcord
    temp[0,:] = temp[0,:]/temp[2,:]
    temp[1,:] = temp[1,:]/temp[2,:]
    pts1 = np.array([temp[0,:],temp[1,:]]).T
    pts1[:,[0,1]] = pts1[:,[1,0]]
    
    temp2 = np.linalg.pinv(H_new/H_new.max())@hdcord
    temp2[0,:] = temp2[0,:]/temp2[2,:]
    temp2[1,:] = temp2[1,:]/temp2[2,:]
    pts2 = np.array([temp2[0,:],temp2[1,:]]).T
    pts2[:,[0,1]] = pts2[:,[1,0]]   

    M,v = mean_var_error(ptso,pts1)
    M_LM,v_LM = mean_var_error(ptso,pts2)
    
    
    for i in range (ptso.shape [0]):        
        pointo = (int(ptso[i,0].item()),int(ptso[i,1].item())) #Actual Point
        point1 = (int(pts1[i,0].item()),int(pts1[i,1].item())) #Points without any refinement
        point2 = (int(pts2[i,0].item()),int(pts2[i,1].item())) #Points with refinement
        
        pimg = cv2.circle(pimg,pointo,radius =3,color=(0,0,255),thickness = -1)
        pimg = cv2.putText(pimg,str(i),pointo,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
        
        pimg = cv2.circle(pimg,point1,radius =3,color=(0,255,0),thickness = -1)
        pimg = cv2.putText(pimg,str(i),point1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
        
        pimg = cv2.circle(pimg,point2,radius =3,color=(255,0,0),thickness = -1)
        pimg = cv2.putText(pimg,str(i),point2,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
        
    plt.imshow(pimg)
    plt.axis('off')
    return M,v,M_LM,v_LM
    


# In[16]:


#Converts R to Rodrigues representation
def parameterized_comp(k,R_all,t_all):
    K = np.asarray([k[0,0],k[0,1],k[0,2],k[1,1],k[1,2]])
    phi = np.arccos((np.trace(R_all.T)-1)/2)
    wee = (phi/(2*np.sin(phi))*np.array([R_all[:,2,1]-R_all[:,1,2],R_all[:,0,2]-R_all[:,2,0],R_all[:,1,0]-R_all[:,0,1]])).T
    W = wee.flatten() 
    Wt = np.append(W,t_all.flatten())
    ans = np.append(K,Wt)
    
    return ans
    
    


# In[17]:


#Converts from Rodrigues representation to R
def deparameterized_comp(ans,n_img=30):
    k = ans[0:5]
    K = np.array([[k[0],k[1],k[2]],[0,k[3],k[4]],[0,0,1]])
    
    w = ans[5:3*n_img+5]
    w = w.reshape(n_img,-1)
    I = np.eye(3,3)
    phi = np.linalg.norm(w,axis=1)
    zero = np.zeros((n_img))
    
    W33X = np.array([[zero,-w[:,2],w[:,1]],[w[:,2],zero ,-w[:,0]],[-w[:,1],w[:,0],zero ]])
    W3X = np.einsum('ijk->kij', W33X)
    Re = (np.sin(phi)/phi)[:, None, None]*W3X + ((1-np.cos(phi))/(phi**2))[:, None, None]*np.matmul(W3X,W3X)
    R = Re+I 
    t = ans[3*n_img+5:]
    T = t.reshape(n_img,-1)
    
    return K,R,T
    
    
    


# In[18]:


#Cost function for Zhang LM
def cost_Func_Zhang(ans,dcoord,rcoord):
    K,R,t = deparameterized_comp(ans)
    cost_all=[]
    for i in range(R.shape[0]):
        rg = R[i]
        tg = t[i]
        RTG = np.concatenate([rg[:,0:1],rg[:,1:2],tg.reshape((-1,1))],axis=1)
        H_g = np.dot(K,RTG)
        H = H_g/H_g.max()
        
        r = rcoord[i]        
        hdcoord = np.insert(dcoord,2,1,axis=1)
        hrcoord = np.insert(r,2,1,axis=1)
    
        hrcoord_calc = np.matmul(H,hdcoord.T)
        hrcoord_calc = hrcoord_calc/hrcoord_calc[2,:]
        hrcoord_calc = hrcoord_calc.T
        cost  = np.abs(hrcoord-hrcoord_calc)
        cost= cost.sum(axis=1)**2
        cost_all.append(cost)
        
    final_cost = np.array(cost_all)
    return final_cost.flatten()


# In[19]:


#Inbuilt LM function
def LM_inbuilt_Zhang(pts,K1,R1,T1):
    xx,yy = np.meshgrid(np.arange(0,10),np.arange(0,8))
    rcoords = np.vstack((xx.reshape(-1),yy.reshape(-1))).T    
    ansy = parameterized_comp(K1,R1,T1)
    LM = least_squares(cost_Func_Zhang,ansy,args=(rcoords*10,pts),method="lm")
    p_LM = LM.x
    K_new,R_new,t_new = deparameterized_comp(p_LM)
    return K_new,R_new,t_new


# In[20]:


#Generating all the homographies
H_all = []
listi = []
inter_pts_nonull = []
all_pts = []
for i in range(40):
    img = cv2.imread("HW8-Files/Dataset1/Pic_"+str(i+1)+".jpg")
    H,pts = Homography_wrapper(img,plot=False)
    if H is not None:
        listi.append(i+1)
        inter_pts_nonull.append(pts)
    H_all.append(H)
    all_pts.append(pts)

all_pts_nonull = np.array(inter_pts_nonull)
H_all_nonull = list(filter(lambda item: item is not None, H_all))


# In[21]:


index_nums = [0,1,2,4,5,7,8,10,11,13,14,15,16,18,20,22,23,24,26,28,29,30,31,32,33,34,35,37,38,39]
H_trim = [H_all[val] for val in index_nums]
pts_trim = [all_pts[val] for val in index_nums]
p_trim = np.array(pts_trim)


# In[22]:


#Calculating the calibration parameters
K= Calc_K(H_trim)
t,R = Calc_Rt(H_trim,K)


# In[23]:


#Refining the parameters using LM
K_new,R_new,t_new = LM_inbuilt_Zhang(p_trim,K,R,t)


# In[29]:


a,b,c,d = Plot_points(H_trim[3],23,K,R,t,K_new,R_new,t_new,index_nums)
print("The mean error before LM refinement is",a)
print("The mean error after LM refinement is",c)
print("The variance in error before LM refinement is",b)
print("The variance in error after LM refinement is",d)


# In[28]:


print("The R matrix before LM refinement is\n",R[23])
print("\nThe R matrix after LM refinement is\n",R_new[23])
print("\nThe t vector before LM refinement is\n",t[23])
print("\nThe t vector after LM refinement is\n",t_new[23])


# In[30]:


a,b,c,d = Plot_points(H_trim[3],2,K,R,t,K_new,R_new,t_new,index_nums)
print("The mean error before LM refinement is",a)
print("The mean error after LM refinement is",c)
print("The variance in error before LM refinement is",b)
print("The variance in error after LM refinement is",d)


# In[31]:


print("The R matrix before LM refinement is\n",R[2])
print("\nThe R matrix after LM refinement is\n",R_new[2])
print("\nThe t vector before LM refinement is\n",t[2])
print("\nThe t vector after LM refinement is\n",t_new[2])


# In[55]:


i=10
dimg = cv2.imread("HW8-Files/Dataset1/Pic_"+str(i+1)+".jpg")
H = inter_pts(dimg,True)


# In[35]:


np.set_printoptions(suppress=True)
print("The intrinsic camera caliberation parameter before LM refinement is \n", K)
print("\nThe intrinsic camera caliberation parameter after LM refinement is \n", K_new)


# In[ ]:




