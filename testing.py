# OUT: Data Loaded.
# OUT: 0     s
# OUT: 1     s
# OUT: 2     s
# OUT: 3     s
# OUT: 4     s
# OUT: 5     s
# OUT: 6     s
# OUT: 7     s
# OUT: 8     b
# OUT: 9     s
# OUT: 10    s
# OUT: 11    s
# OUT: 12    s
# OUT: 13    s
# OUT: 14    s
# OUT: ...
# OUT: 549985    s
# OUT: 549986    s
# OUT: 549987    s
# OUT: 549988    s
# OUT: 549989    s
# OUT: 549990    s
# OUT: 549991    s
# OUT: 549992    b
# OUT: 549993    s
# OUT: 549994    s
# OUT: 549995    b
# OUT: 549996    s
# OUT: 549997    s
# OUT: 549998    s
# OUT: 549999    s
# OUT: Name: Class, Length: 550000, dtype: object
df.rank()
# OUT: <class 'pandas.core.frame.DataFrame'>
# OUT: Int64Index: 550000 entries, 0 to 549999
# OUT: Data columns (total 3 columns):
# OUT: EventId    550000  non-null values
# OUT: probs      550000  non-null values
# OUT: Class      550000  non-null values
# OUT: dtypes: float64(3)
df.probs.rank()
# OUT: 0     113449
# OUT: 1     327478
# OUT: 2     162161
# OUT: 3     233396
# OUT: 4     503485
# OUT: 5     170210
# OUT: 6     305326
# OUT: 7     308606
# OUT: 8      76698
# OUT: 9     276595
# OUT: 10    382386
# OUT: 11    160652
# OUT: 12    192246
# OUT: 13    378880
# OUT: 14    218670
# OUT: ...
# OUT: 549985    472077
# OUT: 549986    467441
# OUT: 549987    253954
# OUT: 549988    323204
# OUT: 549989    257013
# OUT: 549990    411267
# OUT: 549991    380641
# OUT: 549992     89246
# OUT: 549993    450441
# OUT: 549994    498221
# OUT: 549995     48450
# OUT: 549996    197833
# OUT: 549997    117908
# OUT: 549998    363388
# OUT: 549999    114054
# OUT: Name: probs, Length: 550000, dtype: float64
df['RankOrder'] = df.probs.rank()
df
# OUT: <class 'pandas.core.frame.DataFrame'>
# OUT: Int64Index: 550000 entries, 0 to 549999
# OUT: Data columns (total 4 columns):
# OUT: EventId      550000  non-null values
# OUT: probs        550000  non-null values
# OUT: Class        550000  non-null values
# OUT: RankOrder    550000  non-null values
# OUT: dtypes: float64(2), int64(1), object(1)
df.to_csv('data/submission.csv',cols=['EventId','RankOrder','Class'],index=False)
