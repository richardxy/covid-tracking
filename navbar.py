import dash_bootstrap_components as dbc


def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("By Region/Country", href="/by_region")),
            dbc.NavItem(dbc.NavLink("Search", href="/search")),
            dbc.NavItem(dbc.NavLink("Model/Predict", href="/model")),
            dbc.NavItem(dbc.NavLink("CADSEA", href="http://www.cadsea.org", external_link=True, target='_blank'))
            # dbc.DropdownMenu(
            #     nav=True,
            #     in_navbar=True,
            #     label="Menu",
            #     children=[
            #         dbc.DropdownMenuItem("Entry 1"),
            #         dbc.DropdownMenuItem("Entry 2"),
            #         dbc.DropdownMenuItem(divider=True),
            #         dbc.DropdownMenuItem("Entry 3"),
            #     ],
            # ),
        ],
        brand="Home",
        brand_href="/home",
        sticky="top",
    )
    return navbar
