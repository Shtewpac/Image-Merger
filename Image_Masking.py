# import the necessary packages
import argparse
import sys
import openai
import cv2
import numpy as np
from PIL import Image
import requests
import glob, os
import psutil
import time

import selectinwindow

openai.api_key = "<your dalle2 api key>"

sys.setrecursionlimit(10 ** 9)

TARGET_DIM = 512

def convertToTranspPNG(img_path):
    img = Image.open(img_path)
    rgba = img.convert("RGBA")
    datas = rgba.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else: newData.append(item)
    rgba.putdata(newData)
    rgba.save(img_path, "PNG")

def cropImg(pathToImg, outPath = "test_imgs/croppedImg.png", inverted = False):
    wName = "select region to be cropped"
    image = cv2.imread(pathToImg)
    dimensions = image.shape
    print("dimensions:",dimensions)
    imageWidth = dimensions[1]
    imageHeight = dimensions[0]
    rectI = selectinwindow.DragRectangle(image, wName, imageWidth, imageHeight)
    cv2.namedWindow(rectI.wname)
    cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
    while True:
        cv2.imshow(wName, rectI.image)
        key = cv2.waitKey(1) & 0xFF
        if rectI.returnflag:
            break
    cv2.destroyAllWindows()
    r = rectI.outRect
    cropped_image = image[r.y:r.y+r.h, r.x:r.x+r.w]
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (r.x,r.y), (r.x+r.w,r.y+r.h), 255, -1)
    if (inverted == True): 
        masked = cv2.bitwise_and(image, image, mask=mask)
        masked2 = cv2.bitwise_xor(image, masked)
        cropped_image = masked2
    # cv2.imshow("cropped", cropped_image)
    filenameOut = outPath
    cv2.imwrite(filenameOut, cropped_image)
    convertToTranspPNG(filenameOut)

def blend_images(img1Path,img2Path,img1Summary,img2Summary,outputImgDimensions = 1024,gapPercent = 0.1,num_generations = 1):
    image1 = Image.open(img1Path)
    image2 = Image.open(img2Path)
    is1 = image1.size
    is2 = image2.size
    sf1 = (outputImgDimensions/is1[1])
    sf2 = (outputImgDimensions/is2[1])
    image1 = image1.resize((int(sf1*is1[0]),int(sf1*is1[1])))
    image2 = image2.resize((int(sf2*is2[0]),int(sf2*is2[1])))
    # image1.show()
    # image2.show()
    image1_size = image1.size
    image2_size = image2.size
    # print('image1_size:',image1_size)
    # print('image2_size:',image2_size)

    gap_size = int(gapPercent*image1_size[0]) # gap between images
    combinedW = image1_size[0] + gap_size + image2_size[0]
    print(combinedW)
    combined_image = Image.new('RGB',(combinedW, image1_size[1]), (0,0,0))
    combined_image.paste(image1,(0,0))
    combined_image.paste(image2,(gap_size + image1_size[0],0))
    combined_path = TEMP_PATH + "/merged_image.png"
    combined_image.save(combined_path,"PNG")
    convertToTranspPNG(combined_path)
    combined_image = Image.open(combined_path)
    combined_image.paste(image1,(0,0))
    combined_image.paste(image2,(gap_size + image1_size[0],0))
    combined_image.save(combined_path,"PNG")
    # combined_image.show()
    cropCenter = image1_size[0]+gap_size/2
    cl = cropCenter - image1_size[1]/2
    cr = cropCenter + image1_size[1]/2

    combined_image_w_rect = cv2.imread(combined_path)
    # print("(cl,0):",(int(cl),0))
    cv2.rectangle(combined_image_w_rect,(int(cl),0),(int(cr),image1_size[1]),(0,255,0),2)
    combined_image_w_rect_path = TEMP_PATH + "/merged_image_w_mask_out.png"
    cv2.imwrite(combined_image_w_rect_path,combined_image_w_rect)
    combined_image_w_rect = Image.open(combined_image_w_rect_path)
    combined_image_w_rect.show()

    image_mask = combined_image.crop((cl,0,cr,image1_size[1]))
    image_mask = image_mask.resize((outputImgDimensions,outputImgDimensions))
    print(image_mask.size)
    mask_path = TEMP_PATH + "/mask_image.png"
    image_mask.save(mask_path,"PNG")
    # image_mask.show()

    response = openai.Image.create_edit(
        image=open(mask_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt= img1Summary+img2Summary,
        n=num_generations,
        size=str(outputImgDimensions) + "x" + str(outputImgDimensions)
    )
    
    genNum = len(os.listdir(GEN_PATH))
    outNum = len(os.listdir(OUTPUT_PATH))
    for r in response['data']:
        image_url = r['url']
        img_data = requests.get(image_url).content
        generated_img_name = GEN_PATH + "/" + GEN_FILENAME + str(genNum) + '.png'
        with open(generated_img_name, 'wb') as handler:
            handler.write(img_data)
        completed_mask = Image.open(generated_img_name)
        completed_mask = completed_mask.resize((image1_size[1],image1_size[1]))
        # completed_mask.show()
        blended_image = Image.new('RGB',(combinedW, image1_size[1]), 0)
        # blended_image.show()
        blended_image.paste(combined_image,(0,0),0)
        blended_image.paste(completed_mask,(int(cl),0),0)
        blended_path = OUTPUT_PATH + "/" + OUTPUT_FILENAME + str(outNum) + ".png"
        destroyAllWindows()
        blended_image.show()
        blended_image.save(blended_path,"PNG")
        genNum += 1
        outNum += 1
        print(image_url)

def destroyAllWindows():
    cv2.destroyAllWindows()
    for proc in psutil.process_iter():
        if proc.name() == "display": proc.kill()

######################### RENAME ALL GENERATED FILES ###########################################
def renameGeneratedFiles(gen_path,gen_filename):
    generatedFiles = os.listdir(gen_path)
    for i in range(len(generatedFiles)): os.rename(gen_path+"/"+generatedFiles[i], gen_path+"/"+gen_filename + str(i+len(generatedFiles)) + ".png")
    for i in range(len(generatedFiles)): os.rename(gen_path+"/"+generatedFiles[i], gen_path+"/"+gen_filename + str(i) + ".png")

######################### RENAME ALL OUTPUT FILES ###########################################
def renameOutputFiles(output_path,output_filename):
    outputFiles = os.listdir(output_path)
    for i in range(len(outputFiles)): os.rename(output_path+"/"+outputFiles[i], output_path+"/"+output_filename + str(i+len(outputFiles)) + ".png")
    for i in range(len(outputFiles)): os.rename(output_path+"/"+outputFiles[i], output_path+"/"+output_filename + str(i) + ".png")



GEN_PATH = "generated_imgs"
OUTPUT_PATH = "output"
TEMP_PATH = "temp_imgs"

GEN_FILENAME = "generatedImg"
OUTPUT_FILENAME = "blendedImg"

IMG1_PATH = "test_imgs/P1362309.jpg"
IMG2_PATH = "test_imgs/P1352273.jpg"

# renameGeneratedFiles(GEN_PATH,GEN_FILENAME)
# renameOutputFiles(GEN_PATH,OUTPUT_FILENAME)
blend_images(IMG1_PATH, IMG2_PATH, "rolling green hills", "", 512, 0.7, 10)

