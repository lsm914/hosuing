\
import os
import io
import math
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta

# ──────────────────────────────────────────────────────────────────────────────
# 환경설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
SERVICE_KEY = os.getenv("ODCLOUD_SERVICE_KEY", "")
KAKAO_KEY = os.getenv("KAKAO_REST_KEY", "")
SEOUL_TZ = timezone(timedelta(hours=9))

ALL_AREA_CODES = [100, 200, 300, 312, 338, 360, 400, 410, 500, 513, 560, 600, 621, 680, 690, 700, 712]

CACHE_DIR = ".cache"
GEOCODE_CACHE = os.path.join(CACHE_DIR, "geocode_cache.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)

def today_ymd() -> str:
    return datetime.now(SEOUL_TZ).strftime("%Y-%m-%d")

def two_weeks_range(end_dt: datetime | None = None):
    if end_dt is None:
        end_dt = datetime.now(SEOUL_TZ)
    start = (end_dt - timedelta(days=13)).strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    return start, end

def to_pyeong(m2: float | int | str) -> float:
    try:
        m2 = float(str(m2).replace(",", ""))
        return m2 * 0.3025
    except Exception:
        return math.nan

def safe_int(x):
    try:
        return int(str(x).replace(",", ""))
    except Exception:
        return 0

def safe_float(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return math.nan

# ──────────────────────────────────────────────────────────────────────────────
# API 호출
# ──────────────────────────────────────────────────────────────────────────────
BASE_DETAIL = "https://api.odcloud.kr/api/ApplyhomeInfoDetailSvc/v1/getAPTLttotPblancDetail"
BASE_MODEL  = "https://api.odcloud.kr/api/ApplyhomeInfoDetailSvc/v1/getAPTLttotPblancMdl"
BASE_CMPET  = "https://api.odcloud.kr/api/ApplyhomeInfoCmpetRtSvc/v1/getAPTLttotPblancCmpet"

def odcloud_get(url, params):
    params = {**params, "serviceKey": SERVICE_KEY}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ODCloud API 실패: {r.status_code} {r.text[:200]}")
    return r.json()

def fetch_detail_by_area(area_code: int, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "page": 1,
        "perPage": 5000,
        "cond[SUBSCRPT_AREA_CODE::EQ]": area_code,
        "cond[RCRIT_PBLANC_DE::LTE]": end_date,
        "cond[RCRIT_PBLANC_DE::GTE]": start_date,
    }
    js = odcloud_get(BASE_DETAIL, params)
    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["HOUSE_MANAGE_NO","HOUSE_NM","HSSPLY_ADRES","TOT_SUPLY_HSHLDCO","RCEPT_BGNDE","MVN_PREARNGE_YM","RCRIT_PBLANC_DE"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    return df[keep]

def fetch_models(house_manage_no: str) -> pd.DataFrame:
    params = {"page": 1, "perPage": 500, "cond[HOUSE_MANAGE_NO::EQ]": house_manage_no}
    js = odcloud_get(BASE_MODEL, params)
    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["HOUSE_MANAGE_NO","MODEL_NO","SUPLY_AR","HOUSE_TY","LTTOT_TOP_AMOUNT","SUPLY_HSHLDCO","SPSPLY_HSHLDCO"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    return df[keep]

def fetch_cmpet(house_manage_no: str) -> pd.DataFrame:
    params = {"page": 1, "perPage": 5000, "cond[HOUSE_MANAGE_NO::EQ]": house_manage_no}
    js = odcloud_get(BASE_CMPET, params)
    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["HOUSE_MANAGE_NO","HOUSE_TY","REQ_CNT","CMPET_RATE","RESIDE_SENM","SUBSCRPT_RANK_CODE"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    return df[keep]

# ──────────────────────────────────────────────────────────────────────────────
# 지오코딩
# ──────────────────────────────────────────────────────────────────────────────
def load_geocode_cache() -> pd.DataFrame:
    if os.path.exists(GEOCODE_CACHE):
        try:
            return pd.read_parquet(GEOCODE_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=["address","lat","lon","provider"])

def save_geocode_cache(df: pd.DataFrame):
    try:
        df.drop_duplicates("address").to_parquet(GEOCODE_CACHE, index=False)
    except Exception:
        pass

def kakao_geocode(addr: str):
    if not KAKAO_KEY:
        return None, None
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    r = requests.get(url, headers=headers, params={"query": addr}, timeout=10)
    if r.status_code != 200:
        return None, None
    docs = r.json().get("documents", [])
    if not docs:
        return None, None
    x = float(docs[0]["x"])  # lon
    y = float(docs[0]["y"])  # lat
    return y, x

def geocode_addresses(addresses: list[str]) -> pd.DataFrame:
    cache = load_geocode_cache()
    cached = set(cache["address"]) if not cache.empty else set()

    new_rows = []
    geolocator = Nominatim(user_agent="applyhome_geocoder")
    rate_limited = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    for addr in addresses:
        if addr in cached:
            continue
        lat, lon = (None, None)
        provider = None
        if KAKAO_KEY:
            lat, lon = kakao_geocode(addr)
            provider = "kakao" if lat and lon else None
        if not lat or not lon:
            try:
                loc = rate_limited(addr)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                    provider = provider or "nominatim"
            except Exception:
                pass
        new_rows.append({"address": addr, "lat": lat, "lon": lon, "provider": provider or ""})

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        out = pd.concat([cache, new_df], ignore_index=True)
        save_geocode_cache(out)
        return out
    return cache

# ──────────────────────────────────────────────────────────────────────────────
# 집계 로직
# ──────────────────────────────────────────────────────────────────────────────
def build_cover_df(start_date: str, end_date: str, area_codes: list[int]) -> pd.DataFrame:
    frames = []
    for code in area_codes:
        df = fetch_detail_by_area(code, start_date, end_date)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["HOUSE_MANAGE_NO","HOUSE_NM","HSSPLY_ADRES","TOT_SUPLY_HSHLDCO","RCEPT_BGNDE","MVN_PREARNGE_YM","RCRIT_PBLANC_DE"])
    cover = pd.concat(frames, ignore_index=True).drop_duplicates("HOUSE_MANAGE_NO")
    return cover

def build_detail_df(cover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hmno, name, addr, tot, rcpt, mvn, notice in cover[[
        "HOUSE_MANAGE_NO","HOUSE_NM","HSSPLY_ADRES","TOT_SUPLY_HSHLDCO","RCEPT_BGNDE","MVN_PREARNGE_YM","RCRIT_PBLANC_DE"
    ]].itertuples(index=False):
        models = fetch_models(hmno)
        cmpet = fetch_cmpet(hmno)
        if models.empty:
            continue
        if cmpet.empty:
            for _, m in models.iterrows():
                rows.append({
                    "주택관리번호": hmno, "모델번호": m.get("MODEL_NO"), "모집공고일": notice,
                    "공급면적": m.get("SUPLY_AR"), "주택형": m.get("HOUSE_TY"), "공급금액": m.get("LTTOT_TOP_AMOUNT"),
                    "일반공급": m.get("SUPLY_HSHLDCO"), "특별공급": m.get("SPSPLY_HSHLDCO"),
                    "접수건수": None, "경쟁률": None, "거주지역": None, "순위": None,
                    "단지명": name, "공급위치": addr
                })
        else:
            for _, m in models.iterrows():
                ty = m.get("HOUSE_TY")
                sub = cmpet[cmpet["HOUSE_TY"] == ty]
                if sub.empty:
                    rows.append({
                        "주택관리번호": hmno, "모델번호": m.get("MODEL_NO"), "모집공고일": notice,
                        "공급면적": m.get("SUPLY_AR"), "주택형": ty, "공급금액": m.get("LTTOT_TOP_AMOUNT"),
                        "일반공급": m.get("SUPLY_HSHLDCO"), "특별공급": m.get("SPSPLY_HSHLDCO"),
                        "접수건수": None, "경쟁률": None, "거주지역": None, "순위": None,
                        "단지명": name, "공급위치": addr
                    })
                else:
                    for _, c in sub.iterrows():
                        rows.append({
                            "주택관리번호": hmno, "모델번호": m.get("MODEL_NO"), "모집공고일": notice,
                            "공급면적": m.get("SUPLY_AR"), "주택형": ty, "공급금액": m.get("LTTOT_TOP_AMOUNT"),
                            "일반공급": m.get("SUPLY_HSHLDCO"), "특별공급": m.get("SPSPLY_HSHLDCO"),
                            "접수건수": c.get("REQ_CNT"), "경쟁률": c.get("CMPET_RATE"),
                            "거주지역": c.get("RESIDE_SENM"), "순위": c.get("SUBSCRPT_RANK_CODE"),
                            "단지명": name, "공급위치": addr
                        })
    return pd.DataFrame(rows)

def build_combine_df(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["단지명","공급위치","공급금액","공급면적","공급세대수","접수건수(1순위)","접수건수(1+2순위)","주택형"])
    out_rows = []
    keys = sorted(set(zip(detail["단지명"], detail["주택형"])))
    for site, ht in keys:
        sub = detail[(detail["단지명"]==site) & (detail["주택형"]==ht)]
        if sub.empty:
            continue
        rank1 = sub[sub["순위"].astype(str)=="1"]["접수건수"].apply(safe_int).sum()
        rank12 = sub[sub["순위"].astype(str).isin(["1","2"])]["접수건수"].apply(safe_int).sum()
        loc = sub.iloc[0]["공급위치"]
        amt = sub.iloc[0]["공급금액"]
        area = sub.iloc[0]["공급면적"]
        gen = safe_int(sub.iloc[0]["일반공급"]); spc = safe_int(sub.iloc[0]["특별공급"])
        total_supply = gen + spc
        out_rows.append({
            "단지명": site, "공급위치": loc, "공급금액": amt, "공급면적": area,
            "공급세대수": total_supply, "접수건수(1순위)": rank1, "접수건수(1+2순위)": rank12, "주택형": ht
        })
    return pd.DataFrame(out_rows)

def build_by_complex_df(combine: pd.DataFrame, detail: pd.DataFrame, cover: pd.DataFrame) -> pd.DataFrame:
    if combine.empty:
        return pd.DataFrame(columns=[
            "번호","단지명","공급위치","주택형","평균공급금액","평단가","공급세대수","접수1순위","접수1+2순위","경쟁률1","경쟁률1+2","모집공고일","입주예정월","입주년도"
        ])
    g = []
    for (site, ht), grp in combine.groupby(["단지명","주택형"], dropna=False):
        grp = grp.copy()
        grp["세대수"] = grp["공급세대수"].apply(float)
        grp["금액"]   = grp["공급금액"].apply(lambda x: float(str(x).replace(",","")) if pd.notna(x) else math.nan)
        grp["면적m2"] = grp["공급면적"].apply(lambda x: float(str(x).replace(",","")) if pd.notna(x) else math.nan)
        grp["면적평"] = grp["면적m2"].apply(to_pyeong)
        grp["평단가"] = grp.apply(lambda r: (r["금액"]/r["면적평"]) if (pd.notna(r["면적평"]) and r["면적평"]>0) else math.nan, axis=1)

        w = grp["세대수"].sum() if grp["세대수"].notna().any() else 0
        if w<=0:
            avg_amt = grp["금액"].mean()
            avg_py  = grp["평단가"].mean()
            sup = grp["세대수"].sum()
            r1 = grp["접수건수(1순위)"].sum()
            r12= grp["접수건수(1+2순위)"].sum()
        else:
            avg_amt = (grp["금액"]*grp["세대수"]).sum()/w
            avg_py  = (grp["평단가"]*grp["세대수"]).sum()/w
            sup = w
            r1 = grp["접수건수(1순위)"].sum()
            r12= grp["접수건수(1+2순위)"].sum()

        notice = None
        sub_d = detail[detail["단지명"]==site]
        if not sub_d.empty:
            notice = str(sub_d.iloc[0]["모집공고일"]) or ""
            if len(str(notice))>=7:
                notice = str(notice)[:7]
        mvn = None
        sub_c = cover[cover["HOUSE_NM"]==site]
        if not sub_c.empty:
            mvn = str(sub_c.iloc[0]["MVN_PREARNGE_YM"] or "")
            if len(mvn)==6:
                mvn = f"{mvn[:4]}-{mvn[4:]}"
        mvn_year = mvn[:4] if isinstance(mvn, str) and len(mvn)>=4 else ""

        g.append({
            "단지명": site, "공급위치": grp.iloc[0]["공급위치"], "주택형": ht,
            "평균공급금액": round(avg_amt) if pd.notna(avg_amt) else None,
            "평단가": round(avg_py,1) if pd.notna(avg_py) else None,
            "공급세대수": int(round(sup)) if pd.notna(sup) else None,
            "접수1순위": int(round(r1)) if pd.notna(r1) else None,
            "접수1+2순위": int(round(r12)) if pd.notna(r12) else None,
            "경쟁률1": round((r1/sup),2) if sup else None,
            "경쟁률1+2": round((r12/sup),2) if sup else None,
            "모집공고일": notice,
            "입주예정월": mvn,
            "입주년도": mvn_year,
        })

    out = pd.DataFrame(g).sort_values(["단지명","주택형"]).reset_index(drop=True)
    numbers = []
    current = 0
    prev = None
    for site in out["단지명"]:
        if site != prev:
            current += 1
            prev = site
        numbers.append(current)
    out.insert(0, "번호", numbers)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 엑셀
# ──────────────────────────────────────────────────────────────────────────────
def build_excel(by_complex: pd.DataFrame, cover: pd.DataFrame, detail: pd.DataFrame, combine: pd.DataFrame) -> bytes:
    import xlsxwriter  # ensure engine present
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        by_complex.to_excel(xw, index=False, sheet_name="단지별청약경쟁률")
        cover.to_excel(xw, index=False, sheet_name="aptlist(cover)")
        detail.to_excel(xw, index=False, sheet_name="단지세부정보")
        combine.to_excel(xw, index=False, sheet_name="combine")

        wb = xw.book
        ws = xw.sheets["단지별청약경쟁률"]
        money_fmt = wb.add_format({"num_format": "#,##0"})
        one_dec   = wb.add_format({"num_format": "#,##0.0"})
        two_dec   = wb.add_format({"num_format": "0.00"})

        # E: 평균공급금액, F: 평단가, J/K: 경쟁률
        ws.set_column("E:E", 12, money_fmt)
        ws.set_column("F:F", 12, one_dec)
        ws.set_column("J:K", 10, two_dec)
        ws.freeze_panes(1, 1)
    buf.seek(0)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Applyhome 주간집계", layout="wide")
st.title("🏢 청약정보(ODCloud) 주간 수집·집계")

if not SERVICE_KEY:
    st.error("ODCLOUD_SERVICE_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")

start_default, end_default = two_weeks_range()
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.text_input("시작일 (YYYY-MM-DD)", value=start_default)
with col2:
    end_date = st.text_input("종료일 (YYYY-MM-DD)", value=end_default)
with col3:
    area_mode = st.selectbox("검색지역", ["전체","직접코드입력"], index=0)

if area_mode == "전체":
    area_codes = ALL_AREA_CODES
else:
    codes = st.text_input("콤마로 구분된 지역코드들", value="100,200")
    try:
        area_codes = [int(x.strip()) for x in codes.split(",") if x.strip()]
    except Exception:
        st.warning("지역코드 파싱 오류. 기본 전체로 대체합니다.")
        area_codes = ALL_AREA_CODES

run = st.button("데이터 수집/집계 실행")

if run:
    with st.spinner("표준 목록(cover) 수집 중…"):
        cover_df = build_cover_df(start_date, end_date, area_codes)
    st.success(f"단지 목록 {len(cover_df)}건")
    st.dataframe(cover_df.head(20))

    with st.spinner("세부/경쟁률 조인(detail) 구축 중…"):
        detail_df = build_detail_df(cover_df)
    st.success(f"세부 행 수 {len(detail_df)}건")

    with st.spinner("(단지×주택형) 요약(combine) 산출 중…"):
        combine_df = build_combine_df(detail_df)
    st.success(f"요약 행 수 {len(combine_df)}건")

    with st.spinner("단지별 가중평균/경쟁률(by_complex) 산출 중…"):
        by_complex_df = build_by_complex_df(combine_df, detail_df, cover_df)
    st.success(f"단지별 행 수 {len(by_complex_df)}건")

    st.subheader("📇 단지 카드")
    years = ["(전체)"] + sorted(list(set(by_complex_df["입주년도"].dropna())))
    sel_year = st.selectbox("입주년도 필터", years)
    cards_df = by_complex_df.copy()
    if sel_year != "(전체)":
        cards_df = cards_df[cards_df["입주년도"]==sel_year]

    for site, grp in cards_df.groupby("단지명"):
        st.markdown(f"### {site}")
        grp = grp.sort_values("주택형").head(5)
        st.dataframe(
            grp[["주택형","공급세대수","평균공급금액","평단가","경쟁률1","경쟁률1+2"]]
            .rename(columns={"평균공급금액":"공급가액"})
        )

    st.subheader("🗺️ 단지 위치 지도")
    addrs = list(set(by_complex_df["공급위치"].dropna()))
    geo_cache = geocode_addresses(addrs)
    geo_map = geo_cache.set_index("address")[["lat","lon"]].to_dict("index")

    map_df = by_complex_df.copy()
    map_df["lat"] = map_df["공급위치"].map(lambda a: geo_map.get(a, {}).get("lat"))
    map_df["lon"] = map_df["공급위치"].map(lambda a: geo_map.get(a, {}).get("lon"))
    map_df = map_df.dropna(subset=["lat","lon"])

    rep = map_df.sort_values(["번호","주택형"]).groupby("단지명").first().reset_index()
    rep["label"] = rep["번호"].astype(str)

    if rep.empty:
        st.info("지오코딩 결과가 없어 지도를 표시할 수 없습니다. 주소 표기를 점검하거나 KAKAO_REST_KEY를 설정하세요.")
    else:
        midpoint = [rep["lat"].mean(), rep["lon"].mean()]
        layer_text = pdk.Layer(
            "TextLayer",
            data=rep,
            get_position='[lon, lat]',
            get_text='label',
            get_size=16,
            get_color='[0, 0, 0, 255]',
        )
        layer_scatter = pdk.Layer(
            "ScatterplotLayer",
            data=rep,
            get_position='[lon, lat]',
            get_radius=60,
            pickable=True,
        )
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=7),
            layers=[layer_scatter, layer_text],
            tooltip={"text": "{단지명}\n번호 {번호}\n{공급위치}"}
        ))

    st.subheader("📥 Excel 산출")
    xbytes = build_excel(by_complex_df, cover_df, detail_df, combine_df)
    st.download_button(
        label="엑셀 다운로드 (단지별청약경쟁률 / cover / detail / combine)",
        data=xbytes,
        file_name=f"applyhome_{today_ymd()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ──────────────────────────────────────────────────────────────────────────────
# [NEW] 엑셀(단지세부정보) → 카드 생성 탭
# ──────────────────────────────────────────────────────────────────────────────
import re

def _num(x, default=0):
    """숫자 파싱: '-', '', '(△52)' 같은 값 안전 처리."""
    if pd.isna(x):
        return default
    s = str(x).strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except:
        return default

def _band_from_house_ty(s: str) -> int:
    """주택형 앞쪽 3자리(숫자)를 band로. 예: '084.8422A' → 84, '101.9980A' → 101"""
    if pd.isna(s):
        return None
    s = str(s).strip()
    m = re.match(r"(\d{2,3})", s)
    if m:
        return int(m.group(1))
    # 백업: 소수점 앞자리
    p = s.split(".")[0]
    m = re.search(r"(\d{2,3})", p)
    return int(m.group(1)) if m else None

def _to_pyeong(m2):
    try:
        return float(m2) * 0.3025
    except:
        return None

def build_cards_from_detail_sheet(detail_df: pd.DataFrame, cover_df: pd.DataFrame|None=None):
    """
    detail_df: '단지세부정보' 시트 DataFrame
    cover_df:  (선택) 'aptlist(cover)' 시트 DataFrame (있으면 입주예정월 표시)
    return: {단지명: {'meta': {...}, 'rows': pd.DataFrame, 'subtotal': dict}}
    """
    df = detail_df.copy()

    # 필드 정규화
    need_cols = ["단지명","주택형","공급금액","공급면적","일반공급","특별공급",
                 "접수건수","순위","모집공고일","공급위치","모델번호"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None

    df["공급금액"]  = df["공급금액"].apply(_num)
    df["공급면적"]  = df["공급면적"].apply(_num)
    df["일반공급"]  = df["일반공급"].apply(_num)
    df["특별공급"]  = df["특별공급"].apply(_num)
    df["접수건수"]  = df["접수건수"].apply(_num)
    df["순위"]      = df["순위"].astype(str).str.strip()
    df["band"]      = df["주택형"].apply(_band_from_house_ty)
    df["면적(평)"]   = df["공급면적"].apply(_to_pyeong)
    df["평단가"]     = df.apply(lambda r: (r["공급금액"]/r["면적(평)"]) if (r["면적(평)"] and r["면적(평)"]>0) else None, axis=1)

    # (단지명, 주택형) 별 공급세대수(=일반+특별) 대표값, 금액/면적 대표값 추출
    # detail 시트는 지역/순위로 행이 중복되므로, 공급/금액/면적은 한 번만 잡아야 가중평균이 올바름
    by_type = (
        df.groupby(["단지명","주택형","band"], dropna=False)
          .agg(공급세대수=("일반공급", "max"))  # 일단 일반공급로 두고 아래서 +특별공급
          .reset_index()
    )
    spc = df.groupby(["단지명","주택형","band"], dropna=False)["특별공급"].max().reset_index(name="특별공급")
    price = df.groupby(["단지명","주택형","band"], dropna=False)[["공급금액","공급면적","면적(평)","평단가"]].max().reset_index()
    by_type = by_type.merge(spc, on=["단지명","주택형","band"], how="left") \
                     .merge(price, on=["단지명","주택형","band"], how="left")
    by_type["세대수"] = (by_type["공급세대수"].fillna(0) + by_type["특별공급"].fillna(0)).astype(float)

    # (단지명, band) 레벨 집계: 금액/평단가 = 세대수 가중평균, 세대수 = 합
    def _wavg(series, weights):
        w = pd.Series(weights).fillna(0).astype(float)
        x = pd.Series(series).astype(float)
        s = (x * w).sum()
        tw = w.sum()
        return (s / tw) if tw > 0 else None

    band_level = (
        by_type.groupby(["단지명","band"], dropna=False)
               .apply(lambda g: pd.Series({
                   "공급세대수": g["세대수"].sum(),
                   "공급가액": _wavg(g["공급금액"], g["세대수"]),
                   "평단가": _wavg(g["평단가"], g["세대수"]),
               }))
               .reset_index()
    )

    # 경쟁률: 원본 df에서 순위별 접수 합 → band로 합산
    rank1 = (df[df["순위"]=="1"]
             .groupby(["단지명","band"])["접수건수"].sum().reset_index(name="접수1"))
    rank12 = (df[df["순위"].isin(["1","2"])]
             .groupby(["단지명","band"])["접수건수"].sum().reset_index(name="접수12"))
    band_level = band_level.merge(rank1, on=["단지명","band"], how="left") \
                           .merge(rank12, on=["단지명","band"], how="left")
    band_level["접수1"]  = band_level["접수1"].fillna(0)
    band_level["접수12"] = band_level["접수12"].fillna(0)
    band_level["경쟁률(1순위)"]   = band_level.apply(lambda r: round(r["접수1"]/r["공급세대수"], 2) if r["공급세대수"] else None, axis=1)
    band_level["경쟁률(1,2순위)"] = band_level.apply(lambda r: round(r["접수12"]/r["공급세대수"], 2) if r["공급세대수"] else None, axis=1)

    # 단지별 메타 (모집공고일: 월단위, 입주예정월은 cover 시트 있으면 조인)
    meta = (df.groupby("단지명")
              .agg(모집공고일=("모집공고일", "first"), 공급위치=("공급위치","first"))
              .reset_index())
    def _to_yyyymm(x):
        if pd.isna(x): return ""
        s = str(x)
        return s[:7] if len(s)>=7 else s

    meta["모집공고월"] = meta["모집공고일"].map(_to_yyyymm)
    if cover_df is not None and "HOUSE_NM" in cover_df.columns:
        cov = cover_df.copy()
        if "MVN_PREARNGE_YM" in cov.columns:
            cov["입주예정월"] = cov["MVN_PREARNGE_YM"].map(lambda v: f"{str(v)[:4]}-{str(v)[4:]}" if (pd.notna(v) and len(str(v))==6) else "")
        else:
            cov["입주예정월"] = ""
        cov = cov.rename(columns={"HOUSE_NM":"단지명"})
        meta = meta.merge(cov[["단지명","입주예정월"]].drop_duplicates("단지명"), on="단지명", how="left")
    else:
        meta["입주예정월"] = ""

    # 단지별 결과 dict
    out = {}
    for site, g in band_level.groupby("단지명"):
        g = g.sort_values("band")
        # 표시 컬럼 구성
        show = g[["band","공급세대수","공급가액","평단가","경쟁률(1순위)","경쟁률(1,2순위)"]].copy()
        show = show.rename(columns={"band":"타입","공급가액":"공급가액(만원)"})
        # 소계(세대수 가중)
        tot_supply = g["공급세대수"].sum()
        tot_amt    = _wavg(g["공급가액"], g["공급세대수"])
        tot_py     = _wavg(g["평단가"], g["공급세대수"])
        tot_r1     = (g["접수1"].sum()/tot_supply) if tot_supply else None
        tot_r12    = (g["접수12"].sum()/tot_supply) if tot_supply else None
        subtotal = {
            "타입": "소계",
            "공급세대수": int(tot_supply) if pd.notna(tot_supply) else None,
            "공급가액(만원)": round(tot_amt) if pd.notna(tot_amt) else None,
            "평단가": round(tot_py) if pd.notna(tot_py) else None,
            "경쟁률(1순위)": round(tot_r1, 2) if tot_r1 is not None else None,
            "경쟁률(1,2순위)": round(tot_r12, 2) if tot_r12 is not None else None,
        }
        # 포맷팅
        def _fmt(df_):
            df_ = df_.copy()
            for c in ["공급세대수","공급가액(만원)"]:
                if c in df_.columns: df_[c] = df_[c].map(lambda v: f"{int(round(v)):,}" if pd.notna(v) else "")
            if "평단가" in df_.columns:
                df_["평단가"] = df_["평단가"].map(lambda v: f"@{int(round(v)):,}" if pd.notna(v) else "")
            for c in ["경쟁률(1순위)","경쟁률(1,2순위)"]:
                if c in df_.columns: df_[c] = df_[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
            return df_

        out[site] = {
            "meta": meta[meta["단지명"]==site].iloc[0].to_dict() if site in set(meta["단지명"]) else {"모집공고월":"","입주예정월":""},
            "rows": _fmt(show),
            "subtotal": subtotal,
        }
    return out

def render_card(site: str, info: dict):
    m = info["meta"]
    rows = info["rows"]
    sub = info["subtotal"]
    st.markdown(
        f"""
        <div style="border:1px solid #666; padding:8px; margin:10px 0; border-radius:6px;">
          <div style="display:flex; justify-content:space-between;">
            <div style="font-weight:700;">{site}</div>
            <div style="font-size:12px; color:#666;">(단위 : 만원)</div>
          </div>
          <div style="font-size:12px; margin-top:6px;">
            <b>모집공고일:</b> {m.get('모집공고월','')} &nbsp;&nbsp;
            <b>입주예정월:</b> {m.get('입주예정월','') or ''}
          </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(rows, use_container_width=True)
    # 소계 줄
    st.markdown(
        f"""
        <div style="display:flex; gap:12px; font-size:13px; margin-top:6px;">
          <div><b>소계</b></div>
          <div>세대수: {sub['공급세대수']:,} </div>
          <div>공급가액: {sub['공급가액(만원)']:,} </div>
          <div>평단가: @{sub['평단가']:,} </div>
          <div>경쟁률(1): {sub['경쟁률(1순위)']:.2f} </div>
          <div>경쟁률(1,2): {sub['경쟁률(1,2순위)']:.2f}</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.header("📥 엑셀(단지세부정보) 업로드 → 카드 생성")
uploaded = st.file_uploader("생성된 Excel(.xlsx)을 올리세요 (3번째 시트: 단지세부정보).", type=["xlsx"])
if uploaded:
    x = pd.ExcelFile(uploaded)
    detail = pd.read_excel(x, sheet_name="단지세부정보")
    cover = pd.read_excel(x, sheet_name="aptlist(cover)") if "aptlist(cover)" in x.sheet_names else None
    cards = build_cards_from_detail_sheet(detail, cover)
    # 단지명 순서 고정: 번호(있다면) or 이름순
    for site in sorted(cards.keys()):
        render_card(site, cards[site])

