import plotly, plotly.plotly as py, plotly.graph_objs as go

plotly.tools.set_credentials_file(username='alfonsocv18', api_key='IukOlHfoQOc9CejJEThc')

class plotController():

    def __init__():
        '''
        Init function
        '''

    def plotBox(plot_data, msg):
        trace = go.Box(y=plot_data)
        data = [trace]
        layout = dict(
            title = msg
        )
        fig = dict(data=data, layout=layout)
        py.plot(fig)

    def scatterPlot(x, y, msg):
        trace = go.Scatter(x = x,y = y,mode = 'markers')
        layout = dict(title=msg)
        data = [trace]
        fig = dict(data=data, layout=layout)
        py.plot(fig)
