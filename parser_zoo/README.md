## Welcome to ParserZoo !

Each chart derendering model possesses a different format, making it challenging to conduct quantitative evaluations across models. To address this issue, we defined a `standard format` to facilitate easier evaluation and training. Additionally, we developed a parser zoo to convert various formats into our `standard format`.

The `standard format` is as follows:
```json
{
    "title": "",
    "xtitle": "",
    "ytitle": "",
    "xmin":"",
    "xmax":"",
    "ymin":"",
    "ymax":"",
    "data": [
        {
        "name": "",
        "type": "",
        "points": [
            [0,0],
            [0,0],
            ...
        ]
        },
        {
        "name": "",
        "type": "",
        "points": [
            [0,0],
            [0,0],
            ...
        ]
        }
    ]
   
}
```