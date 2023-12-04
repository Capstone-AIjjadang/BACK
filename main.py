from fastapi import FastAPI, Form,File, UploadFile, HTTPException
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import requests
import json
from datetime import datetime
import hmac
import hashlib
from pytz import timezone
from fastapi import Query
from sqlalchemy.orm import relationship
import os
import shutil

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()



SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#권장섭취량 
class Recommended_Intake(Base):
     __tablename__ = "recommended_intake"
     id = Column(Integer, primary_key=True, index=True)
     recommended_cal = Column(Float)
     recommended_nat = Column(Float)
     recommended_carbs = Column(Float)
     recommended_protein = Column(Float)
     recommended_fat = Column(Float)
     recommended__potassium = Column(Float)
    


#일일동안 먹은 총 섭취 영양소(현재 섭취량)
class DayTotalSum(Base):
    __tablename__ = "day_total_sum"

    id = Column(Integer, primary_key=True, index=True)
    Total_food_cal = Column(Float)
    Total_food_nat = Column(Float)
    Total_food_carbs = Column(Float)
    Total_food_protein = Column(Float)
    Total_food_fat = Column(Float)
    Total_food_potassium = Column(Float)

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
    Total_food_potassium = Column(Float)

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
    food_potassium = Column(Float)
    



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
    

    ##time_infos = relationship("TimeInfo", back_populates="user")
    ##all_food_infos = relationship("AllFoodInfo", back_populates="user")
    ##nutrition_infos = relationship("NutritionInfo", back_populates="user")
   ## text_infos = relationship("TextInfo", back_populates="user")
    ##textimage_infos = relationship("TextimageInfo", back_populates="user")
    ##food_infos = relationship("FoodInfo", back_populates="user")
    ##foodimage_infos = relationship("FoodimageInfo", back_populates="user")


#개인정보 테이블 추가
class UserInfo(Base):
    __tablename__ = "user_info"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String)
    medical_history = Column(Text)


#영양성분 테이블 추가
class NutritionInfo(Base):
    __tablename__ = "nutrition_info"

    id = Column(Integer, primary_key=True, index=True)
    calories = Column(Float)
    protein = Column(Float)
    carbohydrates = Column(Float)
    fat = Column(Float)
    sodium = Column(Float)
    potassium = Column(Float)



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




#개인정보 api 
@app.post("/submit_user_join/")
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


#개인정보 api 
@app.post("/submit/")
async def submit_info(
    age: int = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    gender: str = Form(...),
    medical_history: str = Form(...),
):
    db_data = UserInfo(age=age, weight=weight, height=height, gender=gender, medical_history=medical_history)

    db = SessionLocal()
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    db.close()

    response_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "gender": gender,
        "medical_history": medical_history,
    }

    print("Received data:", response_data)

    return response_data

#영양성분 api
@app.post("/submit_nutrition_info/")
async def submit_nutrition_info(
    calories: float = Form(...),
    protein: float = Form(...),
    carbohydrates: float = Form(...),
    fat: float = Form(...),
    sodium: float = Form(...),
    potassium: float = Form(...),
):
    db_data = NutritionInfo(
        calories=calories,
        protein=protein,
        carbohydrates=carbohydrates,
        fat=fat,
        sodium=sodium,
        potassium=potassium,
    )
    db = SessionLocal()
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    db.close()
    response_data = {
        "calories": calories,
        "protein": protein,
        "carbohydrates": carbohydrates,
        "fat": fat,
        "sodium": sodium,
        "potassium": potassium,
    }
    print("Received nutrition data:", response_data)
    return response_data

#개인정보 데이터 호출 api
@app.get("/fetch_data/")
async def fetch_data():
    db = SessionLocal()
    data = db.query(UserInfo).all()
    db.close()
    return data

#영양분 데이터 호출 api
@app.get("/fetch_nutrition_info/")
async def fetch_nutrition_info():
    db = SessionLocal()
    data = db.query(NutritionInfo).all()
    db.close()
    return data



#음식 사진 api 호출하기
# @app.post("/process_image")
# async def process_image(file: UploadFile = File(...), flag: str = "ALL"):
#     timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3]
#     client_id = "glabs_638c223a818794216d1ba2d03f8f395054565ac1b5bc948c9ff6f392195615be"
#     client_secret = "fb66b6fc7ac25cdd55439205994f85b6729c7f400674c3d1acddd007b003c6e4"
#     client_key = "80aa5a78-3ba6-546f-aefb-3aa7a47dfa77"
#     signature = hmac.new(
#         key=client_secret.encode("UTF-8"),
#         msg=f"{client_id}:{timestamp}".encode("UTF-8"),
#         digestmod=hashlib.sha256,
#     ).hexdigest()
#     url = "https://aiapi.genielabs.ai/kt/vision/food"
#     headers = {
#         "Accept": "*/*",
#         "x-client-key": client_key,
#         "x-client-signature": signature,
#         "x-auth-timestamp": timestamp,
#     }
#     fields = {"flag": flag}
#     obj = {"metadata": json.dumps(fields), "media": file.file}
#     response = requests.post(url, headers=headers, files=obj)

#     if response.ok:
#         json_data = json.loads(response.text)
#         code = json_data["code"]
#         data = json_data["data"]
#         prediction_top1 = data[0]["region_0"]["prediction_top1"]
        
#         result = {"code": code, "prediction_top1": prediction_top1}

#         # 터미널에 결과 출력
#         print("Code:", code)
#         print("Prediction Top1:", prediction_top1)

#         return JSONResponse(content=result, status_code=200)
#     else:
#         error_message = f"Error: {response.status_code} - {response.text}"

#         # 터미널에 에러 출력
#         print(error_message)

#         return JSONResponse(content={"error": error_message}, status_code=500)
    
@app.post("/process_image")
async def process_image(file: UploadFile = File(...), flag: str = "ALL"):
   
   # 현재 시각과 클라이언트 정보를 이용하여 서명 생성
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3]
    client_id = "glabs_638c223a818794216d1ba2d03f8f395054565ac1b5bc948c9ff6f392195615be"
    client_secret = "fb66b6fc7ac25cdd55439205994f85b6729c7f400674c3d1acddd007b003c6e4"
    client_key = "80aa5a78-3ba6-546f-aefb-3aa7a47dfa77"
    signature = hmac.new(
        key=client_secret.encode("UTF-8"),
        msg=f"{client_id}:{timestamp}".encode("UTF-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    # 음식 예측 API 호출을 위한 요청 헤더 및 데이터 설정
    url = "https://aiapi.genielabs.ai/kt/vision/food"
    headers = {
        "Accept": "*/*",
        "x-client-key": client_key,
        "x-client-signature": signature,
        "x-auth-timestamp": timestamp,
    }
    fields = {"flag": flag}
    obj = {"metadata": json.dumps(fields), "media": file.file}
    # 음식 예측 API 호출
    response = requests.post(url, headers=headers, files=obj)

  
    if response.ok:
        json_data = json.loads(response.text)
        code = json_data["code"]
        data = json_data["data"]
        prediction_top1 = data[0]["region_0"]["prediction_top1"]

        # 결과를 데이터베이스에 저장
        save_to_database(prediction_top1)

        result = {"code": code, "prediction_top1": prediction_top1}

        # 터미널에 결과 출력
        print("Code:", code)
        print("Prediction Top1:", prediction_top1)

        return JSONResponse(content=result, status_code=200)
    else:
        error_message = f"Error: {response.status_code} - {response.text}"

        # 터미널에 에러 출력
        print(error_message)

        return JSONResponse(content={"error": error_message}, status_code=500)


def save_to_database(prediction_top1):
    # Extract relevant information
    food_name = prediction_top1.get("food_name", "")
    food_cal = prediction_top1.get("food_cal", 0.0)
    food_nat = prediction_top1.get("food_nat", 0.0)
    food_carbs = prediction_top1.get("food_carbs", 0.0)
    food_protein = prediction_top1.get("food_protein", 0.0)
    food_fat = prediction_top1.get("food_fat", 0.0)
    food_potassium = prediction_top1.get("food_potassium", 0.0)
  
    #새로운 데이터베이스 세션 생성
    db = SessionLocal()

    # 음식 사진 결과 저장하는 데이터베이스 생성 및 저장
    db_food_result = FoodImageInfo(
        food_name=food_name,
        food_cal=food_cal,
        food_nat=food_nat,
        food_carbs=food_carbs,
        food_protein=food_protein,
        food_fat=food_fat,
        food_potassium=food_potassium,
       
    )
    db.add(db_food_result)
    db.commit()
    db.refresh(db_food_result)

    #데이터 세션 닫기
    db.close()


# 먹은양*음식성분 결과 엔드포인트
@app.get("/fetch_total_food_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(TotalFoodInfo).all()
    db.close()
    return data

# 이미지 처리 결과를 반환하는 API 엔드포인트
@app.get("/fetch_food_image_info/")
async def  fetch_food_image_info():
    db = SessionLocal()
    data = db.query(FoodImageInfo).all()
    db.close()
    return data

#개인정보 페이지 정보 엔드포인트
@app.get("/fetch_user_join/")
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
@app.put("/update_user/")
async def update_user(
    name: str = Form(...),
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
    user.name=name
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

@app.put("/update_FoodImageInfo/")
async def update_user(
    amount_eaten: float = Form(...),
    
    
):
    db = SessionLocal()
    food_image_info = db.query(FoodImageInfo).order_by(FoodImageInfo.id.desc()).first()
    if food_image_info is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    # 사용자 정보 업데이트
    food_image_info.food_cal *= amount_eaten
    food_image_info.food_nat *= amount_eaten
    food_image_info.food_carbs *= amount_eaten
    food_image_info.food_protein *= amount_eaten
    food_image_info.food_fat *= amount_eaten
    food_image_info.food_potassium *= amount_eaten
    # 데이터베이스에 변경사항 반영
    db.commit()
    db.close()
    # 업데이트된 사용자 정보를 반환
    # 업데이트가 성공했음을 알리는 메시지를 반환

    return {"message": "사용자 정보가 성공적으로 업데이트되었습니다."}

#음식*먹은양 ->새로운 데이터베이스 (음식사진과 ocr 통합) 
@app.post("/submit_total_food_info/")
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
        Total_food_potassium=food_image_info.food_potassium * amount_eaten,
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
            "Total_food_potassium": total_food_data.Total_food_potassium,
        },
    }

    return response_data

##avc


#총 섭취량 엔드포인트
@app.get("/total_food_sum")
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
        Total_food_potassium  = sum(food.Total_food_potassium for food in data),
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
            "Total_food_potassium": day_sum.Total_food_potassium,
        },
      }

     return response_data

#권장섭취량
@app.get("/recommended_intake")
async def recommended_intake():
     db = SessionLocal()
     user_data = db.query(UserJoin).order_by(UserJoin.id.desc()).first()

     height = user_data.height / 100
     age = user_data.age
     weight = user_data.weight
     
     #남자 활동적 식
     recommended_cal = 662 - (9.53 * age) + 1.25 * (15.91 * weight + 539.6 * height)

     

    # 각 항목별 총합을 계산
     recommended = Recommended_Intake(
     recommended_cal=recommended_cal,
    recommended_nat=2300,
      recommended_carbs=recommended_cal*0.65/4,
      recommended_protein=recommended_cal*0.15/4,
      recommended_fat=recommended_cal*0.2/9,
      recommended_potassium=2300,
)
    
      #Reocmmended Intake에 저장
     db.add(recommended)
     db.commit()
     db.refresh(recommended)

     response_data = {
        "message": "Data successfully submitted",
        "recommended_intake_data": {
        
            "Total_food_cal":  recommended.Total_food_cal,
            "Total_food_nat":  recommended.Total_food_nat,
            "Total_food_carbs":  recommended.Total_food_carbs,
            "Total_food_protein":  recommended.Total_food_protein,
            "Total_food_fat":  recommended.Total_food_fat,
            "Total_food_potassium":  recommended.Total_food_potassium,
        },
      }

     return response_data


   

     
