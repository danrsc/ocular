import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.cm import ScalarMappable
import matplotlib
import matplotlib.colors
import matplotlib.cm
from matplotlib.image import imread


__all__ = [
    'surface_nearest_neighbor_input',
    'make_vertex_data',
    'color_rgba_bytes',
    'dual_color_rgba_bytes',
    'cmap_rgba_bytes',
    'comparison_cmap_rgba_bytes',
    'one_hundred_colors']


def surface_nearest_neighbor_input(surface, core_vertices, max_radius_mm, border_max_radius_mm=None):
    """
    Given a surface and a set of core vertices (typically dipole-vertices), this function finds for each surface vertex
    the core vertex to which it is closest. If a surface vertex is further than max_radius_mm from its closest core
    vertex, then it is not matched to any vertex. If border_max_radius_mm is specified, then it should be greater
    than max_radius_mm, and surface vertices which are further from their closest core vertex than max_radius_mm but
    no further than border_max_radius_mm from their closest core vertex, those matches will be returned separately.
    Args:
        surface: An instance of cortex.polyutils.Surface to match with core vertices
        core_vertices: Indices into surface.pts which specify the core vertices
        max_radius_mm: The maximum distance allowed between a core-vertex and a surface vetex for the two to match
        border_max_radius_mm: A distance, larger than max_radius_mm which specifies when a surface vertex should be
            considered 'on the border' of a core vertex

    Returns:
        indices_surface: The indices of surface vertices which are within max_radius_mm of a core vertex.
            A 1d array with shape (num_matches,) with values in the range [0, ..., len(surface.pts))
        indices_core: The indices into core_vertices of the nearest core vertex for each vertex in indices_surface.
            A 1d array with shape (num_matches,) with values in the bounds [0, ..., len(core_vertices))
        indices_border: The indices of surface vertices which are further than max_radius_mm from a core vertex and
            no further than border_max_radius_mm from a core vertex.
            A 1d array with shape (num_border_matches,) with values in the bounds [0, ..., len(surface.pts))
            Only returned in border_max_radius_mm is given
        indices_border_core: The indices into core_vertices of the nearest core vertex for each vertex in
            indices_border.
            A 1d array with shape (num_border_matches,) with values in the inclusive bounds [0, ..., len(core_vertices))
    """
    if border_max_radius_mm is not None:
        if border_max_radius_mm <= max_radius_mm:
            raise ValueError('border_max_radius_mm ({}) <= max_radius_mm ({})'.format(
                border_max_radius_mm, max_radius_mm))
    distances = cdist(surface.pts, surface.pts[core_vertices])
    indices_core = np.argmin(distances, axis=1)
    indices_surface = np.arange(len(surface.pts))
    distances = distances[(indices_surface, indices_core)]
    indicator_radius = distances <= max_radius_mm
    if border_max_radius_mm is not None:
        indicator_border = np.logical_and(np.logical_not(indicator_radius), distances <= border_max_radius_mm)
        return (indices_surface[indicator_radius],
                indices_core[indicator_radius],
                indices_surface[indicator_border],
                indices_core[indicator_border])
    return indices_surface[indicator_radius], indices_core[indicator_radius]


def make_vertex_data(
        data,
        inverse_op,
        surfaces,
        mask=None,
        radius_mm=None,
        border_radius_mm=None,
        fill_value=np.nan,
        dtype=None):
    """
    Constructs an ndarray that has data on the entire surface from an array which has data only at the source-localized
    dipoles to pass into cortex.Vertex
    Args:
        data: The source localized data from which to create the new array. The left hemisphere should be concatenated
            with the right. An array with shape (num_dipoles, ...)
        inverse_op: The inverse-operator associated with the source-localization for this participant
        surfaces: The cortex.polyutils.Surface instances associated with this participant
        mask: An optional mask on data. Data is only taken from locations where the mask is True.
        radius_mm: If specified, the vertices within this distance of the dipole vertices will take the same value
            as the dipole vertex. This makes the dipoles appear larger in the visualization, which can help visibility
        border_radius_mm: The distance form dipoles to the edge of the border. If specified, this value should be
            larger than radius_mm. Vertices within this distance of the dipole vertices will take the same value as
            the dipole vertex, just as with radius_mm. An indicator array will be returned which indicates which
            vertices are further from the the nearest dipole than radius_mm and no further than border_radius_mm
        fill_value: The value to use for vertices which are not dipole vertices and are not within
            radius_mm/border_radius_mm of dipole vertices
        dtype: The data type of the returned array

    Returns:
        vertex_data. A 1d array of shape (num_vertices,)
        indicator_dipole. A boolean 1d array of shape (num_vertices,) which is True where a vertex is a dipole or is
            less than radius_mm from a dipole (when radius_mm is specified)
        indicator_border. A boolean 1d array of shape (num_vertices,) which is True where a vertex is less than
            border_radius_mm from a dipole and where indicator_dipole is False. Only returned when border_radius_mm
            is given.
    """

    if border_radius_mm is not None and radius_mm is None:
        raise ValueError('If border_radius_mm is given, radius_mm must also be given')

    # indices in the input data
    left_indices = np.arange(len(inverse_op['src'][0]['vertno']))
    right_indices = np.arange(len(inverse_op['src'][1]['vertno']))

    if len(data) != len(left_indices) + len(right_indices):
        raise ValueError('data.shape[0] must be the same the number of dipoles in the inverse operator. '
                         'Expected {}, got {}'.format(len(left_indices) + len(right_indices), len(data)))

    # indices in the VertexData
    left_vertices = inverse_op['src'][0]['vertno']
    right_vertices = inverse_op['src'][1]['vertno']

    vertex_data = np.full(
        (len(surfaces[0].pts) + len(surfaces[1].pts),) + data.shape[1:], fill_value=fill_value, dtype=dtype)
    indicator_dipole = np.full(vertex_data.shape, False)
    indicator_border = None
    vertex_mask = None
    if mask is not None:
        vertex_mask = np.full(vertex_data.shape, False)

    if radius_mm is not None:
        if border_radius_mm is not None:
            left_vertices, left_indices_core_vertices, left_border_vertices, left_indices_border_vertices = \
                surface_nearest_neighbor_input(surfaces[0], left_vertices, radius_mm, border_radius_mm)
            left_indices = left_indices[left_indices_core_vertices]
            left_border_indices = left_indices[left_indices_border_vertices]

            right_vertices, right_indices_core_vertices, right_border_vertices, right_indices_border_vertices = \
                surface_nearest_neighbor_input(surfaces[1], right_vertices, radius_mm, border_radius_mm)
            right_indices = right_indices[right_indices_core_vertices]
            right_border_indices = right_indices[right_indices_border_vertices]

            indicator_border = np.full_like(indicator_dipole, False)
            indicator_border[left_border_vertices] = True
            indicator_border[right_border_vertices + len(surfaces[0].pts)] = True
            vertex_data[left_border_vertices] = data[left_border_indices]
            vertex_data[right_border_vertices + len(surfaces[0].pts)] = \
                data[right_border_indices + len(inverse_op['src'][0]['vertno'])]
            if mask is not None:
                vertex_mask[left_border_vertices] = mask[left_border_indices]
                vertex_mask[right_border_vertices + len(surfaces[0].pts)] = \
                    mask[right_border_indices + len(inverse_op['src'][0]['vertno'])]
        else:
            left_vertices, left_indices_core_vertices = surface_nearest_neighbor_input(
                surfaces[0], left_vertices, radius_mm)
            left_indices = left_indices[left_indices_core_vertices]
            right_vertices, right_indices_core_vertices = surface_nearest_neighbor_input(
                surfaces[1], right_vertices, radius_mm)
            right_indices = right_indices[right_indices_core_vertices]

    indicator_dipole[left_vertices] = True
    indicator_dipole[right_vertices + len(surfaces[0].pts)] = True
    vertex_data[left_vertices] = data[left_indices]
    vertex_data[right_vertices + len(surfaces[0].pts)] = data[right_indices + len(inverse_op['src'][0]['vertno'])]
    if mask is not None:
        vertex_mask[left_vertices] = mask[left_indices]
        vertex_mask[right_vertices + len(surfaces[0].pts)] = mask[right_indices + len(inverse_op['src'][0]['vertno'])]
        vertex_data = np.where(vertex_mask, vertex_data, fill_value)
    result = (vertex_data, indicator_dipole)
    if border_radius_mm is not None:
        result += (indicator_border,)
    if vertex_mask is not None:
        result += (vertex_mask,)
    return result


def dual_color_rgba_bytes(data, indicator_color_1, indicator_color_2, colors=None, background=None):
    """
    Maps data to colors using a dual-color scheme. For each value in data, the value is first converted into a
    dual-color index by taking round(data) % (len(colors) ** 2). This index is then unraveled to give
    index_color_1, index_color_2 for each value in data. colors[index_color_1] is then used as the color wherever
    indicator_color_1 is True, colors[index_color_2] is used wherever indicator_color_2 is True (and indicator_color_1
    is False), and background is used everywhere else. When used in combination with make_vertex_data with a
    border_radius_mm specified, this can create dipole visualizations which have an inner color and a border color.
    That can be useful for plotting a large number of clusters, for example, since the clusters can be distinguished
    from each other by 2 colors instead of just 1.
    Args:
        data: The data to map to colors. If not an integer type, the data will be rounded and converted to an
            integer type
        indicator_color_1: Which points in the data should take color-1
        indicator_color_2: Which points in the data should take color-2
        colors: The color palette to draw from. The number of patterns available is len(colors) ** 2. If not specified,
            defaults to the matplotlib color cycle (i.e. TABLEAU_COLORS), which has 10 separate colors (100 patterns)
        background: The color for the background. If not specified, defaults to (0, 0, 0, 0)

    Returns:
        rgba_bytes: An array with shape data.shape + (4,) that has type uint8 with values ranging from 0-255.
    """
    if colors is None:
        colors = matplotlib.colors.TABLEAU_COLORS

    colors = np.array(list(matplotlib.colors.to_rgba(c) for c in colors))
    background = matplotlib.colors.to_rgba('none') if background is None else matplotlib.colors.to_rgba(background)

    if indicator_color_2 is None:
        indicator_foreground = indicator_color_1
    else:
        indicator_foreground = np.logical_or(indicator_color_1, indicator_color_2)
    c = np.where(indicator_foreground, data, 0)

    if not np.issubdtype(c.dtype, np.integer):
        c = np.round(c).astype(int)
    if indicator_color_2 is None:
        c = np.mod(c, len(colors))
    else:
        c = np.mod(c, len(colors) * len(colors))
        c1, c2 = np.unravel_index(c, (len(colors), len(colors)))
        c = np.where(indicator_color_1, c1, c2)

    result = np.where(
        np.reshape(indicator_foreground, indicator_foreground.shape + (1,)),
        np.reshape(colors[np.reshape(c, -1)], c.shape + (4,)),
        np.reshape(background, (1,) * len(indicator_foreground.shape) + np.shape(background)))

    return (result * 255).astype(np.uint8)


def color_rgba_bytes(data, indicator_foreground=None, colors=None, background=None):
    """
    Maps data to colors. For each value in data, the value is converted into a color index by taking
    round(data) % len(colors). colors[index_color] is then used as the color wherever
    indicator_foreground is True, and background is used everywhere else.
    Args:
        data: The data to map to colors. If not an integer type, the data will be rounded and converted to an
            integer type
        indicator_foreground: If specified, any data where indicator_foreground is False will take the background
            color (or use alpha=0 if background is None). np.isnan(data) is also put in the background
        colors: The color palette to draw from. If not specified,
            defaults to the matplotlib color cycle (i.e. TABLEAU_COLORS), which has 10 separate colors
        background: The color for the background. If not specified, defaults to (0, 0, 0, 0)

    Returns:
        rgba_bytes: An array with shape data.shape + (4,) that has type uint8 with values ranging from 0-255.
    """
    if indicator_foreground is not None:
        indicator_foreground = np.logical_and(indicator_foreground, np.logical_not(np.isnan(data)))
    else:
        indicator_foreground = np.logical_not(np.isnan(data))

    return dual_color_rgba_bytes(data, indicator_foreground, None, colors=colors, background=background)


def comparison_cmap_rgba_bytes(
        data_x,
        data_y,
        cmap=None,
        indicator_foreground=None,
        vmin_x=None,
        vmax_x=None,
        vmin_y=None,
        vmax_y=None,
        background=None):
    """
    Similar to pycortex.Vertex2D behavior, this function allows the comparison of 2 sets of data using a 2D colormap.
    Returns the rgba data corresponding to the 2D colormap specified.
    Args:
        data_x: The data which goes on the x-axis of the 2D colormap
        data_y: The data which goes on the y-axis of the 2D colormap
        cmap: The colormap to use. If None, defaults to pycortex.options.config.get('basic', 'default_cmap2D')
        indicator_foreground: A mask with True where the colormap should be used and False where a background color
            should be used. If None, becomes np.logical_not(np.logical_or(np.isnan(data_1), np.isnan(data_2)))
        vmin_x: The minimum value for colormap normalization along the x-axis.
            If None, vmin_x is determined from the data
        vmax_x: The maximum value to use for colormap normalization along the x-axis.
            If None, vmax_x is determined from the data
        vmin_y: The minimum value for colormap normalization along the y-axis.
            If None, vmin_y is determined from the data
        vmax_y: The maximum value to use for colormap normalization along the y-axis.
            If None, vmax_y is determined from the data
        background: The color for data which is nan or where indicator_foreground is False. If None, defaults to
            (0, 0, 0, 0)

    Returns:
        rgba_bytes: An array with shape data.shape + (4,) that has type uint8 with values ranging from 0-255.
        norm_x: An instance of matplotlib.colors.Normalize that can be used to access information such as vmin_x, vmax_x
        norm_y: An instance of matplotlib.colors.Normalize that can be used to access informatino such as vmin_y, vmax_y
        cmap: The 2D colormap as rgba floats, i.e. in the interval [0, 1]. This cmap can be plotted using imshow, e.g.
    """

    if not np.array_equal(data_x.shape, data_y.shape):
        raise ValueError('data_1 shape ({}) must be equal to data_2 shape ({})'.format(data_x.shape, data_y.shape))

    if cmap is None or isinstance(cmap, str):
        from cortex import options
        if cmap is None:
            cmap = options.config.get('basic', 'default_cmap2D')
        cmap_dir = options.config.get('webgl', 'colormaps')
        color_maps = glob.glob(os.path.join(cmap_dir, '*.png'))
        color_maps = dict(((os.path.split(c)[1][:-len('.png')], c) for c in color_maps))
        if cmap not in color_maps:
            raise ValueError('Unknown color map: {}'.format(cmap))

        cmap = imread(os.path.join(cmap_dir, '{}.png'.format(cmap)))

    if indicator_foreground is not None:
        indicator_foreground = np.logical_and(
            indicator_foreground,
            np.logical_not(np.logical_or(np.isnan(data_x), np.isnan(data_y))))
    else:
        indicator_foreground = np.logical_not(np.logical_or(np.isnan(data_x), np.isnan(data_y)))

    norm_x = matplotlib.colors.Normalize(vmin=vmin_x, vmax=vmax_x, clip=True)
    norm_y = matplotlib.colors.Normalize(vmin=vmin_y, vmax=vmax_y, clip=True)

    min_value_x = 0 if np.count_nonzero(indicator_foreground) == 0 else np.nanmin(data_x)
    min_value_y = 0 if np.count_nonzero(indicator_foreground) == 0 else np.nanmin(data_y)

    data_x = norm_x(np.where(indicator_foreground, data_x, min_value_x))
    data_y = 1 - norm_y(np.where(indicator_foreground, data_y, min_value_y))

    data_x = np.round(data_x * (cmap.shape[1] - 1)).astype(np.uint32)
    data_y = np.round(data_y * (cmap.shape[0] - 1)).astype(np.uint32)

    rgba = np.reshape(cmap[(np.reshape(data_y, -1), np.reshape(data_x, -1))], data_x.shape + (4,))
    rgba = (rgba * 255).astype(np.uint8)

    if background is None:
        background = 'none'

    background = (255 * np.array(matplotlib.colors.to_rgba(background))).astype(np.uint8)
    rgba = np.where(
        np.reshape(indicator_foreground, indicator_foreground.shape + (1,)),
        rgba,
        np.reshape(background, (1,) * len(indicator_foreground.shape) + np.shape(background)))

    return rgba, norm_x, norm_y, cmap


def plot_2d_colormap(
        fig, ax, cmap, vmin_x, vmax_x, vmin_y, vmax_y, x_label=None, y_label=None, use_black_background=False):
    ax.imshow(cmap)
    ax.set_xticks([0, cmap.shape[1]])
    ax.set_yticks([0, cmap.shape[0]])
    ax.set_xticklabels([vmin_x, vmax_x])
    ax.set_yticklabels([vmin_y, vmax_y])
    if x_label is not None:
        ax.set_xlabel(x_label, color='white') if use_black_background else ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label, color='white') if use_black_background else ax.set_ylabel(y_label)
    if use_black_background:
        fig.patch.set_facecolor('black')
        ax.tick_params(color='white', labelcolor='white')


def cmap_rgba_bytes(data, indicator_foreground=None, cmap=None, vmin=None, vmax=None, background=None):
    """
    Maps data to colors using a colormap.
    Args:
        data: The data to map to colors.
        indicator_foreground: If specified, any data where indicator_foreground is False will take the background
            color (or use alpha=0 if background is None). np.isnan(data) is also put in the background
        cmap: The colormap to use. If None, the default matplotlib colormap will be used
        vmin: The minimum value for colormap normalization. If None, the value is determined from the data.
        vmax: The maximum value for colormap normalization. If None, the value is determined from the data.
        background: The color for data which is nan or where indicator_foreground is False. If None, defaults to
            (0, 0, 0, 0)

    Returns:
        rgba_bytes: An array with shape data.shape + (4,) that has type uint8 with values ranging from 0-255.
        mappable: An instance of matplotlib.cm.ScalarMappable that can be used to get information about vmin, vmax, etc.
    """
    if indicator_foreground is not None:
        indicator_foreground = np.logical_and(indicator_foreground, np.logical_not(np.isnan(data)))
    else:
        indicator_foreground = np.logical_not(np.isnan(data))

    if background is None:
        background = 'none'

    background = (255 * np.array(matplotlib.colors.to_rgba(background))).astype(np.uint8)

    mappable = ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)

    min_value = 0 if np.count_nonzero(indicator_foreground) == 0 else np.nanmin(data)
    rgba = mappable.to_rgba(np.where(indicator_foreground, data, min_value), bytes=True)
    rgba = np.where(
        np.reshape(indicator_foreground, indicator_foreground.shape + (1,)),
        rgba,
        np.reshape(background, (1,) * len(indicator_foreground.shape) + np.shape(background)))

    return rgba, mappable


def one_hundred_colors():
    return [
        '#ffcdd2',
        '#ef5350',
        '#d32f2f',
        '#b71c1c',
        '#F48FB1',
        '#D81B60',
        '#AD1457',
        '#880E4F',
        '#BA68C8',
        '#9C27B0',
        '#7B1FA2',
        '#4A148C',
        '#B39DDB',
        '#7E57C2',
        '#512DA8',
        '#5C6BC0',
        '#3F51B5',
        '#303F9F',
        '#64B5F6',
        '#2196F3',
        '#1976D2',
        '#0D47A1',
        '#80DEEA',
        '#00BCD4',
        '#0097A7',
        '#006064',
        '#A5D6A7',
        '#66BB6A',
        '#43A047',
        '#18FFFF',
        '#1DE9B6',
        '#76FF03',
        '#FFEB3B',
        '#FBC02D',
        '#F9A825',
        '#FF3D00',
        '#F57F17',
        '#6D4C41',
        '#4E342E',
        '#9E9D24',
        '#D7CCC8',
        '#3E2723',
        '#F5F5F5',
        '#E0E0E0',
        '#BDBDBD',
        '#9E9E9E',
        '#616161',
        '#424242',
        '#fb5e79',
        '#B0BEC5',
        '#90A4AE',
        '#607D8B',
        '#455A64',
        '#D500F9',
        '#6200EA',
        '#00C853',
        '#B3E5FC',
        '#DCEDC8',
        '#2E7D32',
        '#263238',
        '#D84315',
        '#5c415d',
        '#aa767c',
        '#63223c',
        '#7177a7',
        '#bf8033',
        '#CFD8DC',
        '#37474F',
        '#FFFF00',
        '#8FBC8B',
        '#BDB76B',
        '#F5DEB3',
        '#CD853F',
        '#2F4F4F',
        '#B0E0E6',
        '#000080',
        '#DDA0DD',
        '#D8BFD8',
        '#9F8E6D',
        '#551919',
        '#B1E7CC',
        '#82E7B7',
        '#B76345',
        '#844731',
        '#FF8308',
        '#AEFF07',
        '#014EFF',
        '#4B01FF',
        '#FF0094',
        '#CEFFA8',
        '#FF2101',
        '#5B3B4A',
        '#FEA7AC',
        '#776E66',
        '#C4B6A5',
        '#978C7F',
        '#C1EDDC',
        '#46FE03',
        '#31B002',
        '#BBFE01'
    ]
