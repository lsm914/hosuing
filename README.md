# Applyhome 주간 수집·집계 · Streamlit + Excel 자동화

ODCloud(국토부 청약정보) API에서 **전국(전체) + 최근 2주** 데이터를 수집하여
- Streamlit 대시보드로 **카드(타입·세대수·공급가액·평단가·경쟁률)** 와 **지도(단지 위치 번호)** 를 표시
- `단지별청약경쟁률 / aptlist(cover) / 단지세부정보 / combine` 4개 시트로 구성된 **엑셀 파일**을 다운로드
- GitHub Actions로 **매주 월요일 08:00 KST** 자동으로 최신 엑셀 산출

## 1) 시작하기

```bash
pip install -r requirements.txt
cp .env.example .env   # 서비스키 세팅
# .env 파일을 열어 ODCLOUD_SERVICE_KEY 에 본인 키를 넣어주세요.
# (선택) KAKAO_REST_KEY 를 넣으면 지오코딩 정확도가 올라갑니다.
streamlit run app.py
```

브라우저에서 버튼을 눌러 **수집/집계 실행** → 카드/지도 확인 → **엑셀 다운로드** 버튼으로 결과 파일 저장.

## 2) 환경변수

- `ODCLOUD_SERVICE_KEY` : ODCloud 서비스키(인코딩된 채로 사용)
- `KAKAO_REST_KEY` : (선택) 카카오 로컬 API REST 키

## 3) 스트림릿 화면 기능

- 기간 기본값: **오늘 기준 최근 2주**
- 검색지역: **전체** (필요 시 직접 코드 입력 가능)
- 카드: 단지명별 상위 5개 **주택형**에 대해 `타입·공급세대수·공급가액·평단가·경쟁률(1)·경쟁률(1+2)` 표기
- 지도: 단지별 대표 좌표에 **번호 라벨** 표시(좌표는 Kakao→Nominatim 순으로 지오코딩, 캐시는 `.cache/geocode_cache.parquet`)

## 4) 주간 자동 엑셀 (GitHub Actions)

- 리포지토리 **Settings → Secrets and variables → Actions → New repository secret** 에 아래를 추가:
  - `ODCLOUD_SERVICE_KEY`
  - (선택) `KAKAO_REST_KEY`
- 워크플로우는 **매주 월요일 08:00 (KST)** 에 `applyhome_YYYY-MM-DD.xlsx` 파일을 생성해 **Artifacts**로 업로드합니다.
- 수동 실행은 **Actions → workflow_dispatch → Run workflow** 로 가능합니다.

## 5) 엑셀 시트

- `단지별청약경쟁률` : 핵심 결과(세대수 가중평균 공급가액/평단가, 경쟁률(1/1+2), 모집공고일, 입주예정월, 입주년도, 번호)
- `aptlist(cover)` : 목록(관리번호/단지명/공급위치/세대수/접수일/입주예정월/모집공고일)
- `단지세부정보` : 모델·경쟁률 조인 결과(원천 상세)
- `combine` : (단지×주택형) 요약(접수 1/1+2 합계 등)

## 6) 라이선스
MIT
