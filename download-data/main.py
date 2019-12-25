import yfinance as yf
import sqlalchemy
import uuid
import math

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DATE, DECIMAL
from pandas import DataFrame as df
import logging;

logging.basicConfig(level=logging.INFO)

engine = create_engine('mysql://root:123456@127.0.0.1:3306/stock', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class StockAmerica(Base):
    __tablename__ = 'stock_america'

    id = Column(String, name="uuid", primary_key=True)
    name = Column(String)
    open = Column(DECIMAL)
    high = Column(DECIMAL)
    low = Column(DECIMAL)
    close = Column(DECIMAL)
    volume = Column(Integer)
    adj_close = Column(DECIMAL)
    # dividends = Column(DECIMAL)
    # stock_splits = Column(Integer)
    tran_date = Column(DATE)
    
    def __init__(self, id: object, name: object, open: object, high: object, low: object, close: object, volume: object, adj_close: object, tran_date: object) -> object:
        self.id = id
        self.name = name
        self.open = 0.00 if math.isnan(open) else open
        self.high = 0.00 if math.isnan(high) else high
        self.low = 0.00 if math.isnan(low) else low
        self.close = 0.00 if math.isnan(close) else close
        self.volume = 0 if math.isnan(volume) else volume
        self.adj_close = 0.00 if math.isnan(adj_close) else adj_close
        # self.dividends = dividends
        # self.stock_splits = stock_splits
        self.tran_date = tran_date


class StockAmericaService(object):

    def __init__(self):
        self.session = Session();

    def add_list(self, df, name):
        default_values = {"Open" : 0.00, "High" : 0.00, "Low" : 0.00, "Close" : 0.00, "Volume" : 0, "Adj Close" : 0.00}
        # df.dropna(axis=0, how="all")

        stock_one_list = []
        for index, row in df.iterrows():
            stock_one = StockAmerica(uuid.uuid4(), name, row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Adj Close'], index)
            stock_one_list.append(stock_one)

        logging.info("current stock-%s is saving to db." % name)
        try:
            self.session.add_all(stock_one_list)
            self.session.commit()
        except Exception as e:
            logging.error("save to db error ",exc_info = True)


def main():
    stocks = "A \
 AAL \
 AAP \
 AAPL \
 ABBV \
 ABC \
 ABT \
 ACN \
 ADBE \
 ADI \
 ADM \
 ADP \
 ADS \
 ADSK \
 AEE \
 AEP \
 AES \
 AET \
 AFL \
 AGN \
 AIG \
 AIV \
 AIZ \
 AJG \
 AKAM \
 ALB \
 ALGN \
 ALK \
 ALL \
 ALLE \
 ALXN \
 AMAT \
 AMD \
 AME \
 AMG \
 AMGN \
 AMP \
 AMT \
 AMZN \
 ANDV \
 ANSS \
 ANTM \
 AON \
 AOS \
 APA \
 APC \
 APD \
 APH \
 ARE \
 ARNC \
 ATVI \
 AVB \
 AVGO \
 AVY \
 AWK \
 AXP \
 AYI \
 AZO \
 BA \
 BAC \
 BAX \
 BBT \
 BBY \
 BCR \
 BDX \
 BEN \
 BIIB \
 BK \
 BLK \
 BLL \
 BMY \
 BSX \
 BWA \
 BXP \
 C \
 CA \
 CAG \
 CAH \
 CAT \
 CB \
 CBG \
 CBOE \
 CBS \
 CCI \
 CCL \
 CELG \
 CERN \
 CF \
 CFG \
 CHD \
 CHK \
 CHRW \
 CHTR \
 CI \
 CINF \
 CL \
 CLX \
 CMA \
 CMCSA \
 CME \
 CMG \
 CMI \
 CMS \
 CNC \
 CNP \
 COF \
 COG \
 COL \
 COO \
 COP \
 COST \
 COTY \
 CPB \
 CRM \
 CSCO \
 CSRA \
 CSX \
 CTAS \
 CTL \
 CTSH \
 CTXS \
 CVS \
 CVX \
 CXO \
 D \
 DAL \
 DE \
 DFS \
 DG \
 DGX \
 DHI \
 DHR \
 DIS \
 DISCA \
 DISCK \
 DISH \
 DLR \
 DLTR \
 DOV \
 DPS \
 DRE \
 DRI \
 DTE \
 DUK \
 DVA \
 DVN \
 EA \
 EBAY \
 ECL \
 ED \
 EFX \
 EIX \
 EL \
 EMN \
 EMR \
 EOG \
 EQIX \
 EQR \
 EQT \
 ES \
 ESRX \
 ESS \
 ETFC \
 ETN \
 ETR \
 EVHC \
 EW \
 EXC \
 EXPD \
 EXPE \
 EXR \
 F \
 FAST \
 FB \
 FBHS \
 FCX \
 FDX \
 FE \
 FFIV \
 FIS \
 FISV \
 FITB \
 FL \
 FLIR \
 FLR \
 FLS \
 FMC \
 FOX \
 FOXA \
 FRT \
 FTI \
 GD \
 GE \
 GGP \
 GILD \
 GIS \
 GLW \
 GM \
 GOOG \
 GOOGL \
 GPC \
 GPN \
 GPS \
 GRMN \
 GS \
 GT \
 GWW \
 HAL \
 HAS \
 HBAN \
 HBI \
 HCA \
 HCN \
 HCP \
 HD \
 HES \
 HIG \
 HLT \
 HOG \
 HOLX \
 HON \
 HP \
 HPE \
 HPQ \
 HRB \
 HRL \
 HRS \
 HSIC \
 HST \
 HSY \
 HUM \
 IBM \
 ICE \
 IDXX \
 IFF \
 ILMN \
 INCY \
 INFO \
 INTC \
 INTU \
 IP \
 IPG \
 IR \
 IRM \
 ISRG \
 IT \
 ITW \
 IVZ \
 JBHT \
 JCI \
 JEC \
 JNJ \
 JNPR \
 JPM \
 JWN \
 K \
 KEY \
 KHC \
 KIM \
 KLAC \
 KMB \
 KMI \
 KMX \
 KO \
 KORS \
 KR \
 KSS \
 KSU \
 L \
 LB \
 LEG \
 LEN \
 LH \
 LKQ \
 LLL \
 LLY \
 LMT \
 LNC \
 LNT \
 LOW \
 LRCX \
 LUK \
 LUV \
 LVLT \
 LYB \
 M \
 MA \
 MAA \
 MAC \
 MAR \
 MAS \
 MAT \
 MCD \
 MCHP \
 MCK \
 MCO \
 MDLZ \
 MDT \
 MET \
 MGM \
 MHK \
 MKC \
 MLM \
 MMC \
 MNST \
 MO \
 MON \
 MOS \
 MPC \
 MRK \
 MRO \
 MS \
 MSFT \
 MSI \
 MTB \
 MTD \
 MU \
 MYL \
 NAVI \
 NBL \
 NDAQ \
 NEE \
 NEM \
 NFLX \
 NFX \
 NI \
 NKE \
 NLSN \
 NOC \
 NOV \
 NRG \
 NSC \
 NTAP \
 NTRS \
 NUE \
 NVDA \
 NWL \
 NWS \
 NWSA \
 O \
 OKE \
 OMC \
 ORCL \
 ORLY \
 OXY \
 PAYX \
 PBCT \
 PCAR \
 PCG \
 PDCO \
 PEG \
 PEP \
 PFE \
 PFG \
 PG \
 PGR \
 PH \
 PHM \
 PKG \
 PKI \
 PLD \
 PM \
 PNC \
 PNR \
 PNW \
 PPG \
 PPL \
 PRGO \
 PRU \
 PSA \
 PSX \
 PVH \
 PWR \
 PX \
 PXD \
 PYPL \
 QCOM \
 QRVO \
 RCL \
 RE \
 REG \
 REGN \
 RF \
 RHI \
 RHT \
 RJF \
 RL \
 RMD \
 ROK \
 ROP \
 ROST \
 RRC \
 RSG \
 RTN \
 SBAC \
 SBUX \
 SCG \
 SCHW \
 SEE \
 SHW \
 SIG \
 SJM \
 SLB \
 SLG \
 SNA \
 SNI \
 SNPS \
 SO \
 SPG \
 SPLS \
 SRCL \
 SRE \
 STI \
 STT \
 STX \
 STZ \
 SWK \
 SWKS \
 SYF \
 SYK \
 SYMC \
 SYY \
 T \
 TAP \
 TDG \
 TEL \
 TGT \
 TIF \
 TJX \
 TMK \
 TMO \
 TRIP \
 TROW \
 TRV \
 TSCO \
 TSN \
 TSS \
 TWX \
 TXN \
 TXT \
 UAA \
 UAL \
 UDR \
 UHS \
 ULTA \
 UNH \
 UNM \
 UNP \
 UPS \
 URI \
 USB \
 UTX \
 V \
 VAR \
 VFC \
 VIAB \
 VLO \
 VMC \
 VNO \
 VRSK \
 VRSN \
 VRTX \
 VTR \
 VZ \
 WAT \
 WBA \
 WDC \
 WEC \
 WFC \
 WHR \
 WM \
 WMB \
 WMT \
 WRK \
 WU \
 WY \
 WYN \
 WYNN \
 XEC \
 XEL \
 XL \
 XLNX \
 XOM \
 XRAY \
 XRX \
 XYL \
 YUM \
 ZBH \
 ZION \
 ZTS"

    st_list = stocks.split()

    # # stocks_format_str = " ".join(st_list)
    # stocks_format_str = "A AA AAPL"
    # data = yf.download(stocks_format_str, period="2m", group_by = 'ticker')
    # i = 0;
    # while i < len(st_list):
    #     temp = i + 2
    #     if temp >= len(st_list) - 1:
    #         temp = len(st_list) - 1
    #     temp_list = st_list[i:temp]
    #     stocks_format_str = " ".join(temp_list)
    #     try:
    #         data = yf.download(stocks_format_str, period="max", group_by='ticker', threads=True)
    #
    #         for stock_name in temp_list:
    #             stockAmericaService = StockAmericaService()
    #             stockAmericaService.add_list(data[stock_name], stock_name)
    #     except Exception as e:
    #         logging.error("Invoke data from http error", exc_info=True)

    try:
        for stock in st_list:
            spy_index_stocks_info = yf.download(stock, period="max", group_by='ticker',threads=8)
            stockAmericaService = StockAmericaService()
            stockAmericaService.add_list(spy_index_stocks_info, stock)
    except Exception as e:
        logging.error("Invoke data from http error",exc_info = True)

if __name__ == '__main__':
    main();
    # stockAmericaService = StockAmericaService()
    # albb = yf.Ticker("BABA");
    # albb_info = albb.history(start='2018-01-01', end='2018-01-30');
    # stockAmericaService.add_list(albb_info, "BABA")
    # print(albb_info)