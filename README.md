# 👁 Littlerome AI Vision

ระบบตรวจจับความผิดปกติของท่อส่งก๊าซและท่อน้ำ ด้วย **YOLOv11** แบบ Real-time  
พัฒนาด้วย Streamlit · บันทึกข้อมูลลง Google Sheets · รองรับ Multi-user

---

## 🎯 Detection Classes

| Class | ความหมาย | สี |
|-------|---------|-----|
| `pipeline_legit` ✅ | ท่อปกติ ไม่มีความผิดปกติ | เขียว |
| `water_leak` 💧 | น้ำรั่วซึม | น้ำเงิน |
| `pipe_crack` 🔧 | ท่อแตก / ร้าว | ส้ม |
| `flame` 🔥 | เพลิงไหม้ / ไฟลุก | แดง |
| `corrosion` ⚠️ | การกัดกร่อน | ม่วง |
| `rust` 🟫 | สนิม | ส้มเข้ม |

---

## 🗂 โครงสร้าง Repository

```
littlerome/
├── app.py                          ← Streamlit app หลัก
├── Corrosion_Best_Model.pt         ← YOLOv11 model
├── PipeCrack.pt                    ← YOLOv11 model
├── Pipeline_Legit_Best_Model.pt    ← YOLOv11 model
├── water_leak_best.pt              ← YOLOv11 model
├── Flame_Best_Model.pt             ← YOLOv11 model      
├── Rust.pt                         ← YOLOv11 model
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System dependencies (Streamlit Cloud)
├── README.md
└── .streamlit/
    └── config.toml                 ← Dark theme config
```

---

## 🚀 Deploy บน Streamlit Cloud

### 1. Push ไฟล์ขึ้น GitHub
```bash
git add app.py Flame_Best_Model.pt water_leak_best.pt Pipeline_Legit_Best_Model.pt PipeCrack.pt Corrosion_Best_Model.pt Rust.pt requirements.txt packages.txt README.md
git add .streamlit/config.toml
git commit -m "init: Littlerome AI Vision"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main
```

### 2. Deploy
1. ไป [share.streamlit.io](https://share.streamlit.io) → **New app**
2. เลือก repo → Main file: `app.py`
3. กด **Deploy**

### 3. ตั้ง Secrets
ไปที่ **Settings → Secrets** วางข้อความนี้:

```toml
invite_code = "YOUR_INVITE_CODE"

[gsheets]
sheet_id = "YOUR_GOOGLE_SHEET_ID"

[gcp_service_account]
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = """-----BEGIN RSA PRIVATE KEY-----
... วาง private key ทั้งก้อน ...
-----END RSA PRIVATE KEY-----
"""
client_email = "xxx@xxx.iam.gserviceaccount.com"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

---

## 💻 รันในเครื่อง

```bash
pip install -r requirements.txt
streamlit run app.py
```

สำหรับ Google Sheets ในเครื่อง สร้างไฟล์ `.streamlit/secrets.toml` ตามรูปแบบด้านบน

---

## ☁️ ตั้ง Google Sheets

1. ไป [console.cloud.google.com](https://console.cloud.google.com) → สร้าง Project
2. เปิด **Google Sheets API** + **Google Drive API**
3. สร้าง **Service Account** → ดาวน์โหลด JSON key
4. สร้าง Google Sheet ใหม่ → copy **Sheet ID** จาก URL
5. Share Sheet ให้ `client_email` จาก JSON → Role: **Editor**
6. นำข้อมูลจาก JSON ใส่ใน Streamlit Secrets

Google Sheets จะมี 2 sheet อัตโนมัติ:

| Sheet | เก็บอะไร |
|-------|---------|
| `History` | ประวัติการ detect ทุกครั้ง |
| `Users` | บัญชีผู้ใช้ทั้งหมด |

---

> ⚠️ **เปลี่ยน password ทันทีหลัง deploy**

### Roles
| Role | สิทธิ์ |
|------|-------|
| `admin` | ทำได้ทุกอย่าง + จัดการ user |
| `user` | ใช้งาน detect ได้ทุก feature |
| `viewer` | ดูได้อย่างเดียว |

### สมัครสมาชิก
- ต้องมี **Invite Code** จาก Admin
- ตั้ง Invite Code ได้ใน Secrets: `invite_code = "..."`
- Default invite code: `lt5600`

---

## 🗂 Features

| Tab | รายละเอียด |
|-----|-----------|
| 📊 **Dashboard** | KPI cards, defect breakdown, system status, recent inspections |
| 📷 **Live Camera** | Webcam detect + อัดวิดีโอ + export GIF/MP4 |
| 🖼 **Upload Image** | วิเคราะห์รูป + download ผลพร้อม bounding box |
| 🎬 **Upload Video** | Frame-by-frame analysis + timeline summary |
| 📋 **History** | ประวัติบันทึกใน Google Sheets + export CSV |
| ⚙️ **Settings** | Confidence threshold, alerts, user management |

---

## 📦 Dependencies

```
streamlit>=1.35.0
ultralytics>=8.2.0
opencv-python-headless>=4.9.0
Pillow>=10.3.0
numpy>=1.26.0
pandas>=2.2.0
gspread>=6.0.0
google-auth>=2.28.0
```

---

## 📄 License

Littlerome AI Vision © 2025 · YOLOv11 by [Ultralytics](https://ultralytics.com)
