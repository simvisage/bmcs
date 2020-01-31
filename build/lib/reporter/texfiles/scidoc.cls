\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{lib/scidoc}[2013/07/16 Arnd Koeppe - Scientific Document]

% CLASS OPTIONS
\DeclareOption{monochrome}{\PassOptionsToPackage{\CurrentOption}{xcolor}}
\DeclareOption*{\ClassWarning{scidoc}{Unknown option '\CurrentOption'}}
\ProcessOptions\relax

% BASE CLASS
\LoadClass[10pt,twoside,openany]{scrbook}

% REQUIRED PACKAGES
\RequirePackage[T1]{fontenc}
\RequirePackage{calc}
\RequirePackage[includeheadfoot]{geometry}
\RequirePackage[headsepline,         %
        plainheadsepline,       %
        footsepline,         % 
        plainfootsepline]{scrpage2}

\RequirePackage{xcolor}    % xcolor required before pdfpages
\RequirePackage{pdfpages}  % required before tocstyle
\RequirePackage{tocstyle}  % advanced table of contents
\RequirePackage{microtype}  % advanced typography
%\RequirePackage{titlesec}

% DEFINE MARGINS AND HEAD AND FOOT SIZE (GEOMETRY PACKAGE)
\geometry{paper = a4paper, %
          lmargin = 20mm, %
          rmargin = 20mm, %
          tmargin = 10mm, %23mm,
          bmargin = 12mm, %25mm,
          headheight = 6mm, %
          headsep = 7mm, %
          footskip = 15mm}

% ADVANCED TYPOGRAPHY
\microtypesetup{activate = {true,nocompatibility},   % activate protrusion and expansion
                              %  (go beyond margins for smooth text edge)
                tracking = true,            % activate tracking
                                    %  (modify spaces uniformly between letters)
                kerning = true,              % activate kerning
                                     %  (modify spaces between specific letters)
                spacing = true,              % activate spacing
                                    %  (modify spaces between words)
                factor = 1000,              % allowed protrusion amount (default 1000)
                stretch = 10,              % stretch letters (default 20)
                shrink = 10}              % shrink letters (default 20)
\SetExtraKerning[unit=space]
          {encoding={*}, family={*}, series={*}, size={footnotesize,small,normalsize}}
          {\textendash={400,400},       % en-dash, add more space around it
            "28={ ,150},             % left bracket, add space from right
           "29={150, },             % right bracket, add space from left
            \textquotedblleft={ ,150},     % left quotation mark, space from right
            \textquotedblright={150, }}     % right quotation mark, space from left
\SetExtraKerning[unit=space]
           {encoding={*}, family={*}, series={b}, size={large,Large,huge,Huge}}
           {1={-200,-200},           % slightly negative spacing around 1
           \textendash={400,400}}        % en-dash, add more space around it
\SetTracking{encoding={*}, shape=sc}{0}        % no tracking around small capitals (scshape)
\microtypecontext{spacing = nonfrench}

% PARAGRAPH AND VERTICAL SPACING
\KOMAoptions{parskip=half, 
            headings=small}

% TEXTBODY FONT (KOMA)
\addtokomafont{sectioning}{\rmfamily}    % serifs for headings, titles and toc
\setkomafont{pagenumber}{\slshape}      % slanted page number
\addtokomafont{descriptionlabel}{\rmfamily}  % serifs for description items

% HEADER AND FOOTER FORMAT (KOMA)
\pagestyle{scrheadings}              % style of normal pages
\renewcommand*{\titlepagestyle}{blank}      % title page style
\renewcommand*{\partpagestyle}{scrplain}    % part page style
\renewcommand*{\chapterpagestyle}{scrheadings}  % chapter page style
\renewcommand*{\indexpagestyle}{scrheadings}  % index page style

\setheadsepline{0.5pt}  % thickness of header separation line
\setfootsepline{0.5pt}  % thickness of footer separation line

% fix vertical position of page number (pseudo footheight)
%\cfoot[\rule{0pt}{15mm} \pagemark]  % scrplain
%    {\rule{0pt}{15mm} \pagemark}  % scrheadings
\ofoot[\rule{0pt}{6mm} \pagemark]  % scrplain
    {\rule{0pt}{6mm} \pagemark}  % scrheadings


% TITLE FORMAT (KOMA)
\setcounter{secnumdepth}{3}     % number label for subsubsection
\KOMAoptions{numbers = noenddot}  % no dot after disposition numbers
%\renewcommand*{\partformat}{\partname~\thepart.}        % always add dot after part number
%\renewcommand*{\chapterformat} %
%        {\chapappifchapterprefix{\ }\thechapter.\enskip}  % always add dot after chapter number

% TITLE SPACING (TITLESEC)
\newcommand{\chapterheadfont}{}    % font used for chapter head
\renewcommand*{\chaptername}{}

% \titlespacing{\chapter}{0mm}              % indentation
%              {-4.5ex}              % spacing above
%              {9ex}              % spacing below
% \titlespacing{\section}{0mm}              % indentation
%              {1.5ex plus 0.15ex minus 0.15ex}    % spacing above
%              {0.3ex plus 0.05ex minus 0.05ex}  % spacing below
% \titlespacing{\subsection}{0mm}              % indentation
%              {1ex plus 0.1ex minus 0.1ex}      % spacing above
%              {0.1ex plus 0.01ex minus 0.01ex}  % spacing below
% \titlespacing{\subsubsection}{0mm}            % indentation
%              {0.5ex plus 0.05ex minus 0.05ex}      % spacing above
%              {0.1ex plus 0.01ex minus 0.01ex}  % spacing below
% \titlespacing{\paragraph}{0mm}              % indentation
%              {0.01ex}      % spacing above
%              {1em}  % spacing below

% CAPTIONS
\renewcommand*{\captionformat}{\ }  % only space between caption label and text

% TABLE OF CONTENTS
\setcounter{tocdepth}{3}        % add subsubsections to table of contents
\KOMAoptions{bibliography = totoc,  % add bibliography
       listof = totoc}    % add list of figures, tables, etc
\usetocstyle{KOMAlike}        % toc and list of ... format style
\settocfeature{entryvskip}{1.0ex}  % set vertical space between toc, listof, etc entries


% TITLE PAGE SETTINGS
\deftripstyle{blank}[0pt][0.5pt]{ }{ }{ }{ }{ }{ }  % only header and foot lines

% INDEX PAGE STYLE
%\renewenvironment{theindex}
%{\if@twocolumn
%   \@restonecolfalse
% \else
%   \@restonecoltrue
% \fi
% \columnseprule \z@
% \columnsep 35\p@
% \twocolumn[\@makeschapterhead{\indexname}]%
% \@mkboth{\indexname}%
%         {\indexname}%
% \thispagestyle{fancy}\parindent\z@
% \parskip\z@ \@plus .3\p@\relax
% \let\item\@idxitem
% \raggedright}

% HERITAGE COMMANDS
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
