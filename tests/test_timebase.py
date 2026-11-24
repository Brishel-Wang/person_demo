"""单元测试：片内时间换算、分桶。"""
from person_demo.timebase import bucket_index_for, format_local_us, local_us_to_pts, pts_to_local_us


def test_pts_to_local_us_anchor_zero():
    # 60fps: pts_anchor=0, tb=1/60
    assert pts_to_local_us(0, 0, 1, 60) == 0
    assert pts_to_local_us(60, 0, 1, 60) == 1_000_000
    assert pts_to_local_us(30, 0, 1, 60) == 500_000


def test_pts_to_local_us_with_anchor():
    # 锚点非零：第一帧 local_us=0 是片内原点
    assert pts_to_local_us(1000, 1000, 1, 60) == 0
    assert pts_to_local_us(1060, 1000, 1, 60) == 1_000_000


def test_roundtrip():
    assert local_us_to_pts(2_500_000, 1000, 1, 60) == 1150
    assert pts_to_local_us(1150, 1000, 1, 60) == 2_500_000


def test_rational_exactness():
    # 29.97fps time_base 1001/30000
    # 30030 pts * (1001/30000) s = 1002.001 s
    assert pts_to_local_us(30030, 0, 1001, 30000) == 1_002_001_000


def test_format_local_us():
    assert format_local_us(0) == "00:00.000"
    assert format_local_us(61_234_567) == "01:01.234"


def test_buckets_60s():
    assert bucket_index_for(0) == 0
    assert bucket_index_for(59_999_999) == 0
    assert bucket_index_for(60_000_000) == 1
    # [0,200s) = 4 桶
    assert {bucket_index_for(x * 1_000_000) for x in [0, 60, 120, 180]} == {0, 1, 2, 3}
