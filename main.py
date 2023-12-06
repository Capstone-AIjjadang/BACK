from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query, Form
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import vision
from PIL import Image, ImageFont, ImageDraw
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

import base64
import requests
import json
from datetime import datetime
import hmac
import hashlib
from pytz import timezone

import asyncio
import numpy as np
import os
import sys
import io
import platform
import re
import cv2
import shutil
import math
import aiofiles



DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#권장섭취량 
class Recommended_Intake(Base):
    __tablename__ = "recommended_intake"

    id = Column(Integer, primary_key=True, index=True)
    recommended_cal = Column(Float)
    recommended_nat = Column(Integer)
    recommended_carbs = Column(Float)
    recommended_protein = Column(Float)
    recommended_fat = Column(Float)
    

#일일동안 먹은 총 섭취 영양소(현재 섭취량)
class DayTotalSum(Base):
    __tablename__ = "day_total_sum"

    id = Column(Integer, primary_key=True, index=True)
    Total_food_cal = Column(Float)
    Total_food_nat = Column(Float)
    Total_food_carbs = Column(Float)
    Total_food_protein = Column(Float)
    Total_food_fat = Column(Float)

#음식 *먹은양 통합 데이터베이스
class TotalFoodInfo(Base):
    __tablename__ = "food_total_info"

    id = Column(Integer, primary_key=True, index=True)
    Total_food_name = Column(String)
    Total_food_cal = Column(Float)
    Total_food_nat = Column(Float)
    Total_food_carbs = Column(Float)
    Total_food_protein = Column(Float)
    Total_food_fat = Column(Float)

#음식사진 인식 결과 저장 테이블
class FoodImageInfo(Base):
    __tablename__ = "food_image_info"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String)
    food_cal = Column(Float)
    food_nat = Column(Float)
    food_carbs = Column(Float)
    food_protein = Column(Float)
    food_fat = Column(Float)

#성분표 인식 결과 저장 테이블
class TextImageInfo(Base):
    __tablename__ = "text_image_info"

    id = Column(Integer, primary_key=True, index=True)
    text_name = Column(String)
    text_cal = Column(String)
    text_nat = Column(String)
    text_carbs = Column(String)
    text_protein = Column(String)
    text_fat = Column(String)

#성분표 인식 결과 저장 테이블
class OCRImageInfo(Base):
    __tablename__ = "OCR_image_info"

    id = Column(Integer, primary_key=True, index=True)
    text_image_data = Column(LargeBinary)

#음식 인식 사진 저장 테이블
class FImageInfo(Base):
    __tablename__ = "image_of_food"

    id = Column(Integer, primary_key=True, index=True)
    food_image_data = Column(LargeBinary)


Base.metadata.create_all(bind=engine)
#회원정보 테이블 추가
class UserJoin(Base):
    __tablename__ = "userjoin"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    identity = Column(String)
    password = Column(String)
    email = Column(String)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String)
    medical_history = Column(String)
    

#음식추천
class Food:
    def __init__(self, name, weight, energy, carbs, fat, protein, nat):
        self.name = name
        self.weight = weight
        self.energy = energy
        self.carbs = carbs
        self.fat = fat
        self.protein = protein
        self.nat = nat

    @classmethod
    def from_dict(cls, item):
        return cls(
            name=item['음식명'],
            weight=item['g'],
            energy=item['cal'],
            carbs=item['carbs'],
            fat=item['fat'],
            protein=item['protein'],
            nat=item['nat'],
        )

    def calculate_distance(self, target_values, weights):
        if self.nat <= target_values[3]:
            differences = [
                self.carbs - target_values[0],
                self.protein - target_values[1],
                self.fat - target_values[2],
            ]
            distance = math.sqrt(sum(weight * difference**2 for weight, difference in zip(weights, differences)))
            return distance
        else:
            return float('inf')

Base.metadata.create_all(bind=engine)


app = FastAPI()

# CORS 설정
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# #개인정보 데이터 호출 api
@app.get("/today_food_info/")
async def fetch_data():
    db = SessionLocal()
    data = db.query(TotalFoodInfo).all()
    db.close()
    return data


#개인정보 api 
@app.post("/user_join/")
async def submit_join(
    
    name :str=Form(...),
    identity:str=Form(...),
    password :str=Form(...),
    email :str=Form(...),
    age :int=Form(...),
    weight:float=Form(...),
    height:float=Form(...),
    gender :str=Form(...),
    medical_history :str=Form(...),
    
):
    db_data = UserJoin(
        name=name,identity=identity, password=password,email=email, age=age, weight=weight, height=height, 
        gender=gender, medical_history=medical_history)

    db = SessionLocal()
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    db.close()

    response_data = {
        "name":name,
        "identity":identity,
        "password":password,
        "emali":email,
        "age": age,
        "weight": weight,
        "height": height,
        "gender": gender,
        "medical_history": medical_history,
    }

    print("Received data:", response_data)

    return response_data


async def Upload_image(file):
    current_directory = os.path.abspath(os.getcwd())
    directory = os.path.join(current_directory, "temp_images")

    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"{timestamp}_{file.filename}"
    file_location = os.path.join(directory, unique_filename)

    # 파일 비동기적으로 쓰기
    async with aiofiles.open(file_location, 'wb') as buffer:
        file_data = await file.read()
        await buffer.write(file_data)

    # 파일 데이터 동기적으로 읽기 및 데이터베이스에 저장
    with open(file_location, 'rb') as img_file:
        img_data = img_file.read()

        db = SessionLocal()
        try:
            new_image_record = FImageInfo(
                food_image_data=img_data
            )
            db.add(new_image_record)
            db.commit()
            db.refresh(new_image_record)
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()




@app.post("/process_foodimage/")
async def process_image(file: UploadFile = File(...), flag: str = "ALL"):
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3]
    client_id = "glabs_638c223a818794216d1ba2d03f8f395054565ac1b5bc948c9ff6f392195615be"
    client_secret = "fb66b6fc7ac25cdd55439205994f85b6729c7f400674c3d1acddd007b003c6e4"
    client_key = "80aa5a78-3ba6-546f-aefb-3aa7a47dfa77"
    signature = hmac.new(
        key=client_secret.encode("UTF-8"),
        msg=f"{client_id}:{timestamp}".encode("UTF-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    url = "https://aiapi.genielabs.ai/kt/vision/food"
    headers = {
        "Accept": "*/*",
        "x-client-key": client_key,
        "x-client-signature": signature,
        "x-auth-timestamp": timestamp,
    }
    fields = {"flag": flag}
    obj = {"metadata": json.dumps(fields), "media": file.file}

    response = requests.post(url, headers=headers, files=obj)

    if response.ok:
        json_data = json.loads(response.text)
        code = json_data["code"]
        data = json_data["data"]
        prediction_top1 = data[0]["region_0"]["prediction_top1"]

        save_to_database(prediction_top1)

    # 파일 스트림을 처음으로 되돌림
    file.file.seek(0)

    # Upload_image 함수 호출
    await Upload_image(file)
    Upload_image(file)

def save_to_database(prediction_top1):
    food_name = prediction_top1.get("food_name", "")
    food_cal = prediction_top1.get("food_cal", 0.0)
    food_nat = prediction_top1.get("food_nat", 0.0)
    food_carbs = prediction_top1.get("food_carbs", 0.0)
    food_protein = prediction_top1.get("food_protein", 0.0)
    food_fat = prediction_top1.get("food_fat", 0.0)

    db = SessionLocal()
    db_food_result = FoodImageInfo(
        food_name=food_name,
        food_cal=food_cal,
        food_nat=food_nat,
        food_carbs=food_carbs,
        food_protein=food_protein,
        food_fat=food_fat,
    )
    db.add(db_food_result)
    db.commit()
    db.refresh(db_food_result)
    db.close()



# 이미지 처리 결과를 반환하는 API 엔드포인트
@app.get("/food_image_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(FoodImageInfo).all()
    db.close()
    return data

#개인정보 페이지 정보 엔드포인트
@app.get("/user_info/")
async def fetch_user_join():
    db=SessionLocal()
    data=db.query(UserJoin).all()
    user_data = []
    for user in data:
        user_data.append({
            "name": user.name,
            "age": user.age,
            "weight": user.weight,
            "height": user.height,
            "gender": user.gender,
            "medical_history": user.medical_history,
        })
    db.close()
    return user_data

# 사용자 정보를 업데이트하는 PUT 엔드포인트
@app.put("/user_update/")
async def update_user(
    weight: float = Form(...),
    height: float = Form(...),
    gender: str = Form(...),
    medical_history: str = Form(...),
    
):
    db = SessionLocal()
    user = db.query(UserJoin).order_by(UserJoin.id.desc()).first()
    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    # 사용자 정보 업데이트
    user.weight = weight
    user.height = height
    user.gender = gender
    user.medical_history = medical_history

    # 데이터베이스에 변경사항 반영
    db.commit()
    db.close()
    # 업데이트된 사용자 정보를 반환
    # 업데이트가 성공했음을 알리는 메시지를 반환

    return {"message": "사용자 정보가 성공적으로 업데이트되었습니다."}


#음식*먹은양 ->새로운 데이터베이스 (음식사진)
@app.post("/total_food_result/")
async def submit_join(
    amount_eaten: float = Form(...),
    
):
    db = SessionLocal()
    # 가장 최근의 음식 정보를 가져옴
    food_image_info = db.query(FoodImageInfo).order_by(FoodImageInfo.id.desc()).first()

    if not food_image_info:
        raise HTTPException(status_code=404, detail="음식 정보를 찾을 수 없습니다.")

    # TotalFoodInfo에 저장할 데이터 생성
    total_food_data = TotalFoodInfo(
        Total_food_name=food_image_info.food_name,
        Total_food_cal=food_image_info.food_cal * amount_eaten,
        Total_food_nat=food_image_info.food_nat * amount_eaten,
        Total_food_carbs=food_image_info.food_carbs * amount_eaten,
        Total_food_protein=food_image_info.food_protein * amount_eaten,
        Total_food_fat=food_image_info.food_fat * amount_eaten,
    
    )

    # TotalFoodInfo 테이블에 저장
    db.add(total_food_data)
    db.commit()
    db.refresh(total_food_data)

    response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
            "Total_food_name": total_food_data.Total_food_name,
            "Total_food_cal": total_food_data.Total_food_cal,
            "Total_food_nat": total_food_data.Total_food_nat,
            "Total_food_carbs": total_food_data.Total_food_carbs,
            "Total_food_protein": total_food_data.Total_food_protein,
            "Total_food_fat": total_food_data.Total_food_fat,
        
        },
    }

    return response_data

#음식*먹은양 ->새로운 데이터베이스 (ocr )
@app.post("/total_text_result/")
async def submit_join(
    amount_eaten: float = Form(...),
    name: str = Form(...),
):
    db = SessionLocal()
    # 가장 최근의 음식 정보를 가져옴
    food_image_info = db.query(TextImageInfo).order_by(TextImageInfo.id.desc()).first()

    if not food_image_info:
        raise HTTPException(status_code=404, detail="음식 정보를 찾을 수 없습니다.")

    # 문자열을 실수로 변환하는 함수
    def to_float(value):
        try:
            return float(value)
        except ValueError:
            return 0.0

    # TotalFoodInfo에 저장할 데이터 생성
    total_food_data = TotalFoodInfo(
        Total_food_name=name,
        Total_food_cal=to_float(food_image_info.text_cal) * amount_eaten,
        Total_food_nat=to_float(food_image_info.text_nat) * amount_eaten,
        Total_food_carbs=to_float(food_image_info.text_carbs) * amount_eaten,
        Total_food_protein=to_float(food_image_info.text_protein) * amount_eaten,
        Total_food_fat=to_float(food_image_info.text_fat) * amount_eaten,
    )

    # TotalFoodInfo 테이블에 저장
    db.add(total_food_data)
    db.commit()
    db.refresh(total_food_data)

    response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
            "Total_food_name": total_food_data.Total_food_name,
            "Total_food_cal": total_food_data.Total_food_cal,
            "Total_food_nat": total_food_data.Total_food_nat,
            "Total_food_carbs": total_food_data.Total_food_carbs,
            "Total_food_protein": total_food_data.Total_food_protein,
            "Total_food_fat": total_food_data.Total_food_fat,
        },
    }

    return response_data


##avc

##
#총 섭취량 엔드포인트
@app.get("/today_sum_food/")
async def total_food_sum():
    db = SessionLocal()
    data = db.query(TotalFoodInfo).all()

    # 각 항목별 총합을 계산
    day_sum =  DayTotalSum(
        Total_food_cal = sum(food.Total_food_cal for food in data),
        Total_food_nat= sum(food.Total_food_nat for food in data),
        Total_food_carbs  = sum(food.Total_food_carbs for food in data),
        Total_food_protein  = sum(food.Total_food_protein for food in data),
        Total_food_fat = sum(food.Total_food_fat for food in data),
    )
    #TotalFoodInfo에 저장
    db.add(day_sum)
    db.commit()
    db.refresh(day_sum)

    response_data = {
        "message": "Data successfully submitted",
        "total_food_data": {
        
            "Total_food_cal": day_sum.Total_food_cal,
            "Total_food_nat": day_sum.Total_food_nat,
            "Total_food_carbs": day_sum.Total_food_carbs,
            "Total_food_protein": day_sum.Total_food_protein,
            "Total_food_fat": day_sum.Total_food_fat,
        },
    }

    return response_data

#권장섭취량
@app.get("/recommended_intake/")
async def recommended_intake():
    db = SessionLocal()
    user_data = db.query(UserJoin).order_by(UserJoin.id.desc()).first()

    height = user_data.height / 100
    age = user_data.age
    weight = user_data.weight
    gender=user_data.gender
    
    #남자 활동적 식
    #  calo = 662 - (9.53 * age) + 1.25 * ((15.91 * weight) + (539.6 * height))

    print("User data:", gender)
    # 성별에 따라 활동적인 식사량 계산
    if gender == "남자":
        calo = 662 - (9.53 * age) + 1.25 * ((15.91 * weight) + (539.6 * height))
    elif gender == "여자":
        calo = 354 - (6.91 * age) + 1.27 * ((9.36 * weight) + (726 * height))
    else:
        return {"message": "올바른 성별 정보가 제공되지 않았습니다."}     

    # 각 항목별 총합을 계산
    recommended = Recommended_Intake(
    recommended_cal=calo,
    recommended_nat=2300,
    recommended_carbs=calo*0.65/4,
    recommended_protein=calo*0.15/4,
    recommended_fat=calo*0.2/9,
)
    
    #Reocmmended Intake에 저장
    db.add(recommended)
    db.commit()
    db.refresh(recommended)

    response_data = {
    "message": "데이터가 성공적으로 제출되었습니다.",
    "response_data": {
        "recommended_cal": recommended.recommended_cal,
        "recommended_nat": recommended.recommended_nat,
        "recommended_carbs": recommended.recommended_carbs,
        "recommended_protein": recommended.recommended_protein,
        "recommended_fat": recommended.recommended_fat,
    },
}

    return response_data


# 먹은양*음식성분 결과 엔드포인트
@app.get("/list_food_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(TotalFoodInfo).all()
    db.close()
    return data

@app.get("/recommended_food/")
async def submit_join():
    db = SessionLocal()

    # 가장 최근의 음식 정보를 가져옴
    dayTotal_info = db.query(DayTotalSum).order_by(DayTotalSum.id.desc()).first()
    recommended_Intake_info = db.query(Recommended_Intake).order_by(Recommended_Intake.id.desc()).first()

    print("하루합", dayTotal_info.Total_food_carbs)
    print("추천", recommended_Intake_info.recommended_carbs )
    
    # 최근 음식 정보로부터 target_values 계산
    target_values = [
    recommended_Intake_info.recommended_carbs - dayTotal_info.Total_food_carbs,
    recommended_Intake_info.recommended_protein - dayTotal_info.Total_food_protein,
    recommended_Intake_info.recommended_fat - dayTotal_info.Total_food_fat,
    recommended_Intake_info.recommended_nat - dayTotal_info.Total_food_nat
    ]
    print(target_values)

    # JSON 파일 불러오기
    json_path =r"C:\Users\HJ\OneDrive - Sejong University\문서\카카오톡 받은 파일\food.json"
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Food 객체로 변환
    food_list = [Food.from_dict(item) for item in data]

    # 최적의 음식들 찾기
    sorted_foods = find_optimal_foods(food_list, target_values, weights=[1, 1, 1], num_foods=4)

   # 결과 반환
    result = []
    result.append({
            "name1": sorted_foods[0].name,
            "name2": sorted_foods[1].name,
            "name3": sorted_foods[2].name,
            "name4": sorted_foods[3].name
        })
    
    return result

# 음식 유클리드 거리 오름차순 정렬
def find_optimal_foods(food_list, target_values, weights, num_foods=4):
    # 초기화
    best_matches = []

    for food in food_list:
        distance = food.calculate_distance(target_values, weights)
        food.distance = distance
        best_matches.append(food)

    # 거리가 작은 순으로 정렬
    sorted_foods = sorted(best_matches, key=lambda x: x.distance)

    return sorted_foods

import base64

@app.get("/latest_image/")
async def latest_image_info():
    db = SessionLocal()
    try:
        # 가장 최근에 추가된 이미지 가져오기
        latest_image = db.query(OCRImageInfo).order_by(OCRImageInfo.id.desc()).first()
        if latest_image and latest_image.text_image_data:
            # 이미지 데이터를 Base64로 인코딩
            encoded_image = base64.b64encode(latest_image.text_image_data).decode('utf-8')
            return {"image": encoded_image}
        else:
            return {"message": "이미지가 없습니다."}
    finally:
        db.close()

@app.get("/latest_food_image/")
async def latest_food_image_info():
    db = SessionLocal()
    try:
        # 데이터베이스에서 모든 이미지 레코드 가져오기
        all_images = db.query(FImageInfo).all()
        encoded_images = []

        for image in all_images:
            if image.food_image_data:
                encoded_image = base64.b64encode(image.food_image_data).decode('utf-8')
                encoded_images.append({"id": image.id, "image": encoded_image})
            else:
                encoded_images.append({"id": image.id, "image": None})

        return {"images": encoded_images}
    finally:
        db.close()

#OCR 결과 반환
@app.get("/fetch_textimage/")
async def fetch_nutrition_info():
    def convert_image_to_base64(image_binary):
        if image_binary:
            return base64.b64encode(image_binary).decode('utf-8')
        return None

    db = SessionLocal()
    try:
        data = db.query(TextImageInfo).all()
        data2 = db.query(OCRImageInfo).all()

        # TextImageInfo 데이터 직렬화
        textimageinfo_data = [
            {
                "id": item.id, 
                "text_cal": item.text_cal, 
                "text_nat": item.text_nat, 
                "text_carbs": item.text_carbs, 
                "text_protein": item.text_protein, 
                "text_fat": item.text_fat
            } 
            for item in data
        ]

        # OCRImageInfo 데이터 직렬화 (이미지는 Base64로 인코딩)
        timageinfo_data = [{"image": convert_image_to_base64(item.text_image_data)} for item in data2]

        return {"TextImageInfo": textimageinfo_data, "OCRImageInfo": timageinfo_data}
    finally:
        db.close()

#OCR 인식
@app.post("/process_OCRimage") 
async def process_image(file: UploadFile = File(...), flag: str = Query(None)): 
    # 파일 저장 위치 설정 
    current_directory = os.path.abspath(os.getcwd()) 
    directory = os.path.join(current_directory, "temp_images") 

    # 디렉토리가 존재하지 않으면 생성 
    if not os.path.exists(directory): 
        os.makedirs(directory) 

    # 파일 저장 위치 설정 
    file_location = os.path.join(directory, file.filename) 
    with open(file_location, "wb") as buffer: 
        shutil.copyfileobj(file.file, buffer) 

    # OCR 처리 함수 비동기적으로 호출 
    result = await asyncio.to_thread(process_ocr, file_location, flag) 

    # 파일을 바이너리 형태로 읽기
    with open(file_location, "rb") as img_file:
        img_data = img_file.read()

    # 데이터베이스에 이미지 데이터 저장
    db = SessionLocal()
    try:
        new_image_record = OCRImageInfo(
            text_image_data=img_data
        )
        db.add(new_image_record)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

    return {"results": result} 


def process_ocr(file_path, flag): 
    # Google Vision API 클라이언트 설정 
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\toong\Documents\카카오톡 받은 파일\linen-walker-216606-76f54386771c.json"
    client_options = {'api_endpoint': 'eu-vision.googleapis.com'} 
    client = vision.ImageAnnotatorClient(client_options=client_options)  
    # 이미지 파일 읽기 
    with io.open(file_path, 'rb') as image_file: 
        content = image_file.read() 

    image = vision.Image(content=content) 
    response = client.text_detection(image=image) 
    texts = response.text_annotations 

    text_to_display = "" 
    text_output_flag = True 

    for text in texts: 
        if text_output_flag: 
            ocr_text = text.description
            text_to_display += ocr_text + "\n" 
            text_output_flag = False 

    # 함수를 호출하여 결과 얻기 
    nutrition_info = process_nutrition_info(text_to_display) 
    json_data = json.dumps(nutrition_info, indent=4, ensure_ascii=False) 

# 데이터베이스에 결과 저장 
    db = SessionLocal() 
    try: 
        new_record = TextImageInfo( 
            text_cal=nutrition_info.get("kcal", "Unknown"),
            text_nat=nutrition_info.get("나트륨", "Unknown"), 
            text_carbs=nutrition_info.get("탄수화물", "Unknown"), 
            text_protein=nutrition_info.get("단백질", "Unknown"), 
            text_fat=nutrition_info.get("지방", "Unknown"),
        )
        db.add(new_record) 
        db.commit()
    except Exception as e: 
        db.rollback() 
        raise e 
    finally: 
        db.close() 
    return json_data 


def process_nutrition_info(text):
    results = {}

    # 영양소 및 kcal 패턴 정의
    nutrition_pattern = r'(?:(나트륨|탄수화물|지방|단백질)\s+([\d.]+(?:\.[\d.]+)?)\s?(mg|g))'
    kcal_pattern = r'(\d+)\s*kcal'

    # 영양소 매칭
    nutrition_matches = re.findall(nutrition_pattern, text)
    for item_name, ratio, unit in nutrition_matches:
        results[item_name] = f"{ratio} {unit}"

    # kcal 매칭
    kcal_matches = re.findall(kcal_pattern, text)
    if kcal_matches:
        results["kcal"] = f"{kcal_matches[0]} kcal"

    numerical_values = {}
    for key, value in results.items():
        # 숫자 및 소수점 추출
        matches = re.findall(r'[\d.]+', value)
        if matches:
            # 첫 번째 숫자(또는 소수점 숫자)를 저장
            numerical_values[key] = float(matches[0])

    return numerical_values

class Food:
    def __init__(self, name, weight, energy, carbs, fat, protein, nat):
        self.name = name
        self.weight = weight
        self.energy = energy
        self.carbs = carbs
        self.fat = fat
        self.protein = protein
        self.nat = nat

    @classmethod
    def from_dict(cls, item):
        return cls(
            name=item['음식명'],
            weight=item['g'],
            energy=item['cal'],
            carbs=item['carbs'],
            fat=item['fat'],
            protein=item['protein'],
            nat=item['nat'],
        )

    def calculate_distance(self, target_values, weights):
        if self.nat <= target_values[3]:
            differences = [
                self.carbs - target_values[0],
                self.protein - target_values[1],
                self.fat - target_values[2],
            ]
            distance = math.sqrt(sum(weight * difference**2 for weight, difference in zip(weights, differences)))
            return distance
        else:
            return float('inf')
