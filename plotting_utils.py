import os

def save_to_file(x, y, filename, header):
    """Saves x and y data to a file in the tikz_data directory."""
    if not os.path.exists("tikz_data"):
        os.makedirs("tikz_data")
    filepath = os.path.join("tikz_data", filename)
    with open(filepath, 'w') as f:
        f.write(f"# {header}\n")
        f.write("# x y\n")
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]}\n")


def generate_tex_file(figure_name, xlabel, ylabel, plots, xmode='linear', ymode='linear', legend_at='(1,0)', legend_anchor='south east', xmin=None, xmax=None, ymin=None, ymax=None, extra_opts=''):
    """Generates a .tex file for a plot, allowing flexible legend position and anchor."""
    if not os.path.exists("tikz_figures"):
        os.makedirs("tikz_figures")
    filepath = os.path.join("tikz_figures", f"{figure_name}.tex")

    axis_options = [
        "width=14cm",
        "height=9cm",
        "grid=both",
        "grid style={dotted,gray}",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
        "ylabel shift = 0mm",
        f"xmode={xmode}",
        f"ymode={ymode}",
        "axis lines=left",
        "y axis line style={shorten >=-7pt, shorten <=0pt}",
        "legend cell align=left",
        "legend style={font=\\large, draw=none,fill=white}",
        "legend columns=1",
        "legend style={row sep=3pt}",
        "legend style={/tikz/every even column/.append style={column sep=0.4cm}}",
        f"legend style={{at={legend_at},anchor={legend_anchor}}}",
        "grid style={black!40}",
        "ticklabel style = {font=\\large}",
    ]
    if xmin is not None:
        axis_options.append(f"xmin={xmin}")
    if xmax is not None:
        axis_options.append(f"xmax={xmax}")
    if ymin is not None:
        axis_options.append(f"ymin={ymin}")
    if ymax is not None:
        axis_options.append(f"ymax={ymax}")
    if extra_opts:
        axis_options.append(extra_opts)
    
    axis_options_str = ",\n\t".join(axis_options)

    plots_str = ""
    legend_str = ""
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    styles = ["solid", "dashed", "dotted", "dashdot", "dashdotdot"]

    for i, plot in enumerate(plots):
        color = plot.get('color', colors[i % len(colors)])
        style = plot.get('style', styles[i % len(styles)])
        plots_str += f"\\addplot[draw={color}, line width=1.1pt, {style}] table[x index=0,y index=1] {{{{../tikz_data/{plot['data_file']}}}}};\n"
        legend_str += f"\\addlegendentry{{{plot['legend_entry']}}}\n"

    tex_content = f"""\\documentclass{{article}}
\\usepackage[x11names,dvipsnames]{{xcolor}}
\\usepackage{{amsmath}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\usepackage[active,tightpage]{{preview}}
\\PreviewEnvironment{{tikzpicture}}
\\setlength{{\\PreviewBorder}}{{3mm}}
\\pgfplotsset{{compat = newest}}

% for legend
\\pgfplotsset{{
    compat=newest,
    /pgfplots/legend image code/.code={{%
        \\draw[mark repeat=2,mark phase=2,#1]
            plot coordinates {{
                (0cm,0cm)
                (0.6cm,0cm)
                (.8cm,0cm)
            }};
    }},
}}
\\pgfplotsset{{every tick label/.append style={{font=\\large}}}}

\\begin{{document}}

\\input{{./colour_map_AM.tex}}

\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis_options_str}
]

{plots_str}
{legend_str}

\\end{{axis}}
\\end{{tikzpicture}}

\\end{{document}}
"""

    with open(filepath, 'w') as f:
        f.write(tex_content)
    print(f"Generated {filepath}")
