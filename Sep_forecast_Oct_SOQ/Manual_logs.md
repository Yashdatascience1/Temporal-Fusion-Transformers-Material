1. SF table with festive features (as N-1, N-2, etc.) : MOP\_DATABASE.SOQ.FESTIVE\_DATA\_FOR\_DAILY\_FORECASTING
2. SF table with festive features engineered (such as DAYS\_TO\_DIWALI) : MOP\_DATABASE.SOQ.FESTIVE\_FEATURES\_ENGINEERED\_FOR\_DAILY\_FORECASTING
3. Stored procedure to create data for forecasting at daily level : [Stored Procedure](C:\\Users\\G0004878\\Desktop\\TFT_Data\\Daily_forecasting_model\\Iterations in October\\_1_Data preparation\\daily_data_preparation_with_festive_signal_windows.sql)
4. TRAIN\_PERIOD : Apr 1st, 2023 to Apr 30th, 2026 
5. VAL\_PERIOD : May 1st, 2025 to Aug 31st, 2026
6. MODEL\_FAMILY and MODEL\_NAME are same for all rows. Will drop MODEL\_NAME 
7. MODEL\_FAMILY\_CODE is useless, will drop it 
8. FIRST\_SALE\_DATE is not required, will drop it
