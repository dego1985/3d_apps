import pandas as pd
import plotly.express as px


df = pd.read_csv('https://docs.google.com/spreadsheets/d/1dPY1LQAMbR60w246kQV6Iwi4UqmZKgIa4iAlP3wppXo/export?format=csv&gid=995531617')
df = df.rename(columns={
    "What is your order number?": "order_number_str",
    "What is the Model you got?": "model",
    "4G Included?": "4G",
    "CM4 Included?": "CM4",
    "What is the color?": "color",
    "When the shipment notice was sent?":"_notice"
})

df = df.dropna(subset=["_notice"])
df["_notice"] = [x[:-4] + "20"+x[-2:] for x in df["_notice"]]
df["date of notice"] = pd.to_datetime(df["_notice"], format='%m/%d/%Y')

def to_order_number(order_number_str:str):
    order_number_str = order_number_str.replace("x","0")
    order_number_str = order_number_str.replace("X","0")
    return int(order_number_str)
    
def to_order_pattern(model, inc_4G, inc_CM4, color):
    detail = model
    if inc_4G or inc_CM4:
        a = []
        if inc_4G:
            a += ["4G"]
        if inc_CM4:
            a += ["CM4"]
        detail = f"{detail}({','.join(a)})"
    return detail+","+color
df["Order Detail"] = [to_order_pattern(x, y, z, w) for x, y, z, w in zip(df["model"], df["4G"]=="Yes", df["CM4"]=="Yes", df["color"])]
df["Order Number"] = [to_order_number(x) for x in df["order_number_str"]]
df = df[[
    "order number",
    "date of notice",
    "Order detail",
]]

df = df.sort_values("Order detail")
fig = px.scatter(df, x="date of notice", y="order number", color="Order detail")
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()
