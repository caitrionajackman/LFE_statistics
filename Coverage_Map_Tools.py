#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:50:37 2023

@author: bowersch
"""
def create_lobe_mesh():
    import pandas as pd

    import datetime

    import numpy as np

    from trying3 import convert_to_datetime,convert_datetime_to_string, load_MESSENGER_into_tplot,read_in_Weijie_files, check_for_mp_bs_WS,plot_mp_and_bs
    
    #Load in a pandas dataframe with all of the lobe magnetic field data and position
   
    Lobe_Data=pd.read_pickle('Lobe_Data_np.pkl')
    

    # Specify the number of bins in theta and r
    sc=30

    sc2=60
    
    #Specify range in Z to consider

    pos_z=[-4,.5]

    #mesh goes r, theta, z, mag

    mesh_mag=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))
    
    count=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0]))
    
    mesh_eph=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))
    
    mesh_amp_up=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))

    mesh_amp_lp=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))
    
    mesh_sB=np.zeros((sc+1,sc2+1,np.shape(pos_z)[0],3))

    #Specify how far in r and theta you want to go
    distance=5.0

    pos_r=np.arange(sc+1)*distance/(sc)

    distance=np.pi

    pos_theta=np.arange(sc2+1)*distance/(sc2/2)-distance
        
    ephx=Lobe_Data.ephx
    ephy=Lobe_Data.ephy
    ephz=Lobe_Data.ephz
    
    #Define Theta and r for ephemeris data
    theta_d=np.arctan2(ephy,ephx)
            
    r=np.sqrt(ephy**2+ephx**2)
    
        
    magx=Lobe_Data.magx
    
    
    magy=Lobe_Data.magy
    
    magz=Lobe_Data.magz
    
    
    
    magamp=np.sqrt(magx**2+magy**2+magz**2)


    #Create shifted positions to search for data within the bins
    r_pos_r=np.roll(pos_r,-1)

    r_pos_theta=np.roll(pos_theta,-1)

    r_pos_z=np.roll(pos_z,-1)

    for rr in range(np.size(pos_r)):
        print(rr/np.size(pos_r))

        for tt in range(np.size(pos_theta)):
        
            for zz in range(np.size(pos_z)):
            
                gd_a=np.where((r > pos_r[rr]) & (r < r_pos_r[rr]) &\
                            (theta_d > pos_theta[tt]) & (theta_d < r_pos_theta[tt]) &\
                                (ephz > pos_z[zz]) & (ephz < r_pos_z[zz]))[0]
                
                if np.size(gd_a) > 0:
                    
                    l=gd_a[0]
                    
                    #Reassign variable to x,y,z to help keep it straight                              
                    x=rr
                    y=tt
                    z=zz

                        
                    magx_gd=magx.iloc[gd_a]
                    
                    magy_gd=magy.iloc[gd_a]
                    
                    magz_gd=magz.iloc[gd_a]
                    
                    ephx_gd=ephx.iloc[gd_a]
                    
                    ephy_gd=ephy.iloc[gd_a]
                    
                    ephz_gd=ephz.iloc[gd_a]
                    
                    magamp_gd=magamp.iloc[gd_a]
                    
                    sB_gd=Lobe_Data.scaled_B.iloc[gd_a]
                    
                    mesh_mag[x,y,z,0]=np.mean(magx_gd)
                    
                    mesh_sB[x,y,z,0]=np.mean(sB_gd)
                    
                    mesh_mag[x,y,z,1]=np.mean(magy_gd)
                    
                    mesh_mag[x,y,z,2]=np.mean(magz_gd)
                    
                    count[x,y,z]=np.size(gd_a)
                    
                    mesh_eph[x,y,z,0]=np.mean(ephx_gd)
                    
                    mesh_eph[x,y,z,1]=np.mean(ephy_gd)
                    
                    mesh_eph[x,y,z,2]=np.mean(ephz_gd)
                    
                    mesh_amp_up[x,y,z]=np.percentile(magamp_gd,75)
                    
                    mesh_amp_lp[x,y,z]=np.percentile(magamp_gd,25)
                    
    #Save meshes as npy files to call in the plot_lobe_mesh_polar function
    
    np.save('mesh_lobe_up.npy',mesh_amp_up)
    
    np.save('mesh_lobe_lp.npy',mesh_amp_lp)
    
    np.save('mesh_lobe_p.npy',mesh_mag)
    
    np.save('count_lobe_p.npy',count)
    
    np.save('mesh_eph_p.npy',mesh_eph)
    
    np.save('mesh_sB_p.npy',mesh_sB)
    
    np.save('pos_r.npy',pos_r)
    
    np.save('pos_theta.npy',pos_theta)
def plot_lobe_mesh_coverage():
    
    
    #Plots coverage maps of lobe and total orbital segments analyzed (tail)
    import numpy as np
    
    import matplotlib.pyplot as plt
    
    #Load in pos_r and pos_theta 
        
    pos_r=np.load('pos_r.npy')
    
    pos_theta=np.load('pos_theta.npy')
    
    # mesh_tail and count_tail were calculated the same way as mesh_lobe 
    # and count_lobe, but using all of the data not just lobe data
    
    #mesh_tail=np.load('mesh_full_tail.npy')
    
    count_tail=np.load('count_full_tail.npy')
    
    #mesh_lobe=np.load('mesh_lobe_p.npy')
    
    count_lobe=np.load('count_lobe_p.npy')
        
        
    
    fig, axs = plt.subplots(1,3, subplot_kw=dict(projection='polar'))
    
    count_lobe=(count_lobe[:,:,0])
    
    count_tail=count_tail[:,:,0,0]
    
    #Define new color bars for coverage plots
    
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap('viridis', 400000)
    newcolors = viridis(np.linspace(0, 1, 40000))
    pink = np.array([.5])
    newcolors[:1, :] = pink
    newcmp = ListedColormap(newcolors)
    
    viridis = cm.get_cmap('plasma', 400000)
    newcolors = viridis(np.linspace(0, 1, 40000))
    pink = np.array([.5])
    newcolors[:1, :] = pink
    newcmp_r = ListedColormap(newcolors)

    theta,r=np.meshgrid(pos_theta,pos_r)
    
    #Exclue final point because of shading in pcolor meshes
    
    count_lobe=count_lobe[:-1,:-1]
    
    count_tail=count_tail[:-1,:-1]
    
    ratio=count_lobe/count_tail
    
    ratio[count_tail==0]=float('nan')
    
    count_lobe[count_tail<1]=float('nan')
    
    count_tail[count_tail<1]=float('nan')
    

    
    fig0=axs[0].pcolormesh(theta,r,count_lobe/1000.,cmap=newcmp,shading='flat')
    
    
    fig1=axs[1].pcolormesh(theta,r,count_tail/1000.,cmap=newcmp,shading='flat')
    
    fig2=axs[2].pcolormesh(theta,r,ratio,vmin=0,vmax=.8,cmap=newcmp_r,shading='flat')
    
    fig.colorbar(fig0,fraction=0.055,pad=-.1,label='Seconds '+r'$\times$ $10^3$')
    fig.colorbar(fig1,fraction=0.055,pad=-.1,label='Seconds '+r'$\times$ $10^3$')
    
    fig.colorbar(fig2,fraction=0.055,pad=-.1,label='Lobe/Tail')
    
    axs[0].set_title('Lobe Measurements')
    
    axs[1].set_title('Orbital Segments Analyzed')
    
    axs[2].set_title('Ratio')
    
    def format_polar_plots(axis):
        
        
        axis.set_xlim(np.pi/2,3*np.pi/2)
        
        axis.set_xticks(np.linspace(np.pi/2, 3 * np.pi/2, 7))  # Set 6 ticks
        axis.set_xticklabels(['18','20','22','0','2','4','6'])
        
        axis.set_ylim(0,4.5)
        
        axis.set_ylim(0,4.5)
        
        axis.set_ylim(0,4.5)
        
        axis.set_yticks(np.linspace(1,4.0,4))
        
        axis.set_yticklabels(['1 $R_M$','2 $R_M$','3 $R_M$','4 $R_M$'])
        
    format_polar_plots(axs[0])
    format_polar_plots(axs[1])
    format_polar_plots(axs[2])
        

    return