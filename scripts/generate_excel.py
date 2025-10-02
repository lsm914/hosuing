from datetime import datetime, timezone, timedelta
from app import (
    two_weeks_range, build_cover_df, build_detail_df,
    build_combine_df, build_by_complex_df, build_excel, ALL_AREA_CODES
)

def main():
    start, end = two_weeks_range()
    cover = build_cover_df(start, end, ALL_AREA_CODES)
    detail = build_detail_df(cover)
    combine = build_combine_df(detail)
    bycx = build_by_complex_df(combine, detail, cover)
    xbytes = build_excel(bycx, cover, detail, combine)
    fn = f"applyhome_{datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d')}.xlsx"
    with open(fn, "wb") as f:
        f.write(xbytes)
    print("wrote", fn)

if __name__ == "__main__":
    main()
