import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

def datetime(x):
    return np.array(x, dtype=np.datetime64)
def show_plot(x_axis, y_axis_list, y_axis_label_list, plot_title, plot_x_label, plot_y_label, color_list, name="stocks.html"):
    p1 = figure(title=plot_title)
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = plot_x_label
    p1.yaxis.axis_label = plot_y_label
    p1.title.text_font_size = '12pt'

    p1.xaxis.axis_label_text_font_size = '12pt'
    p1.yaxis.axis_label_text_font_size = '12pt'

    for i, curve in enumerate(y_axis_list):
        p1.line(x_axis, curve, line_width=3, color=color_list[i], legend=y_axis_label_list[i])
        # p1.square(x_axis, curve, size=8,  fill_color=None, line_color='#B2DF8A', legend=y_axis_label_list[i])
    p1.legend.location = "center_right"


    output_file(name, title="stocks.py example")

    show(gridplot([[p1]], plot_width=700, plot_height=700))  # open a browser

def show_plot_time(x_axis_list, y_axis_list, y_axis_label_list, plot_title, plot_x_label, plot_y_label, color_list, name='stocks.html'):
    p1 = figure(title=plot_title)
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = plot_x_label
    p1.yaxis.axis_label = plot_y_label
    p1.title.text_font_size = '12pt'

    p1.xaxis.axis_label_text_font_size = '12pt'
    p1.yaxis.axis_label_text_font_size = '12pt'

    for i, curve in enumerate(y_axis_list):
        p1.line(x_axis_list[i], curve, line_width=3, color=color_list[i], legend=y_axis_label_list[i])
        # p1.square(x_axis, curve, size=8,  fill_color=None, line_color='#B2DF8A', legend=y_axis_label_list[i])
    p1.legend.location = "center_right"


    output_file(name, title="stocks.py example")

    show(gridplot([[p1]], plot_width=1500, plot_height=700))  # open a browser