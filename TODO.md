# TODO List

## Weather & Forecasting
- [x] Use weather forecast, not synthetic weather ✅
  - ~~Currently generating synthetic future weather data based on historical patterns~~
  - ✅ Integrated with OpenWeatherMap API for real 5-day forecasts
  - ✅ Uses Meteostat for historical data up to today, then real forecasts from today onwards
  - ✅ Falls back to improved synthetic data when API key is not available
  - ✅ Provides actual predicted air temperatures for more accurate water temperature forecasts

## Future Enhancements
- [ ] Add more sophisticated gap filling algorithms
- [ ] Implement data validation and outlier detection
- [ ] Add export functionality for different file formats
- [ ] Create mobile-responsive design improvements
- [ ] Add user preferences for temperature units (°C/°F)
- [ ] Implement data backup and versioning
- [ ] Add email notifications for extreme temperatures
- [ ] Create historical comparison features (year-over-year)