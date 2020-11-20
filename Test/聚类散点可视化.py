import pandas as pd
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import GeoType
import pickle


fp =  open('./地图可视化.pk', 'rb')
f = open('./城市经纬度.pk', 'rb')
ff = open('./m_list.pk', 'rb')


d = pickle.load(fp)  # {2:class_0, 5: class_1, 8:class_another}
jingweidu = pickle.load(f)  # map_dict
m_list = pickle.load(ff)  # 城市顺序

temp_d = {}

for key, value in d.items():
    for v in value:
        temp_d[m_list[v]] = key

g ={m_list[i]:v for i, v in jingweidu.items()}
print(g)

# style = Style(title_color= "#fff",title_pos = "center",width = 1200,height = 600,background_color = "#404a59")
geo_cities_coords= g
attr=list(temp_d.keys()) 
values= [temp_d[i] for i in attr]
piece=[
      {'min':0, 'max': 3,'label': '类别1','color':'#50A3BA'},  #有上限无下限，label和color自定义
      {'min': 4, 'max': 6,'label': '类别2','color':'#81AE9F'},
      {'min': 7, 'max': 8,'label': '类别3','color':'#E2C568'},
    #   {'min': 150, 'max': 300,'label': '150-300','color':'#FCF84D'},
    #   {'min': 300, 'label': '300以上','color':'#D94E5D'}#有下限无上限
]
# geo = Geo('各个公司位置以及人数',**style.init_style)
# geo = Geo(init_opts=opts.InitOpts())
# geo.add("",attr=attr,value=values,symbol_size= 5,visual_text_color= "#fff",is_piecewise = True,
#         is_visualmap= True,maptype = '河南', 
#         pieces=piece,     #注意，要想pieces生效，必须is_piecewise = True,
#         geo_cities_coords=geo_cities_coords)
# geo.render('聚类.html')





def test_geo():
    city = '长沙'
    g = Geo()
    g.add_schema(maptype='河南')

    # 定义坐标对应的名称，添加到坐标库中 add_coordinate(name, lng, lat)
    # g.add_coordinate('湖南省长沙市宁乡市横市镇藕塘', 112.21369756169062, 28.211359706637378)
    # g.add_coordinate('湖南省长沙市雨花区跳马镇仙峰岭', 113.16921879037058, 28.039877432448428)
    # g.add_coordinate('湖南省长沙市长沙县黄花镇新塘铺长沙黄花国际机场', 113.23212337884058, 28.19327497825815)

    # 定义数据对，
    # data_pair = [('湖南省长沙市雨花区跳马镇仙峰岭', 10), ('湖南省长沙市宁乡市横市镇藕塘', 5), ('湖南省长沙市长沙县黄花镇新塘铺长沙黄花国际机场', 20)]
    data_pair = list(temp_d.items())
    # Geo 图类型，有 scatter, effectScatter, heatmap, lines 4 种，建议使用
    # from pyecharts.globals import GeoType
    # GeoType.GeoType.EFFECT_SCATTER，GeoType.HEATMAP，GeoType.LINES

    # 将数据添加到地图上
    g.add('', data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=15)
    # 设置样式

    g.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    # 自定义分段 color 可以用取色器取色
    # pieces = [
    #     {'max': 1, 'label': '0以下', 'color': '#50A3BA'},
    #     {'min': 1, 'max': 10, 'label': '1-10', 'color': '#3700A4'},
    #     {'min': 10, 'max': 20, 'label': '10-20', 'color': '#81AE9F'},
    #     {'min': 20, 'max': 30, 'label': '20-30', 'color': '#E2C568'},
    #     {'min': 30, 'max': 50, 'label': '30-50', 'color': '#FCF84D'},
    #     {'min': 50, 'max': 100, 'label': '50-100', 'color': '#DD0200'},
    #     {'min': 100, 'max': 200, 'label': '100-200', 'color': '#DD675E'},
    #     {'min': 200, 'label': '200以上', 'color': '#D94E5D'}  # 有下限无上限
    # ]


    pieces=[
      {'min':0, 'max': 3,'label': '类别1','color':'#50A3BA'},  #有上限无下限，label和color自定义
      {'min': 4, 'max': 6,'label': '类别2','color':'#81AE9F'},
      {'min': 7, 'max': 8,'label': '类别3','color':'#E2C568'},
    #   {'min': 150, 'max': 300,'label': '150-300','color':'#FCF84D'},
    #   {'min': 300, 'label': '300以上','color':'#D94E5D'}#有下限无上限
    ]


    #  is_piecewise 是否自定义分段， 变为true 才能生效


    g.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces=pieces),
        title_opts=opts.TitleOpts(title="城市聚类"),
    )


    return g

g = test_geo()
g.render('test_render.html')


