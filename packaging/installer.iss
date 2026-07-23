#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef FileVersion
  #define FileVersion "0.0.0.0"
#endif
#ifndef InstallerBaseName
  #define InstallerBaseName "YOLO数据标注工具箱-dev-windows-x64-setup"
#endif
#ifndef SourceDir
  #define SourceDir "."
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif
#ifndef SourceUrl
  #define SourceUrl ""
#endif

[Setup]
AppId={{7B6C7F86-3F66-44DD-8FC8-6606988679CB}
AppName=YOLO 数据标注工具箱
AppVersion={#AppVersion}
AppVerName=YOLO 数据标注工具箱 {#AppVersion}
AppPublisher=WorkBuddy
#if SourceUrl != ""
AppPublisherURL={#SourceUrl}
AppSupportURL={#SourceUrl}
#endif
DefaultDirName={autopf}\WorkBuddy\YOLO数据标注工具箱
DefaultGroupName=WorkBuddy
DisableProgramGroupPage=yes
AllowNoIcons=yes
LicenseFile={#SourceDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename={#InstallerBaseName}
UninstallDisplayName=YOLO 数据标注工具箱
UninstallDisplayIcon={app}\YOLO工具箱.exe
Uninstallable=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
SetupLogging=yes
CloseApplications=yes
RestartApplications=no
ChangesAssociations=no
ChangesEnvironment=no
VersionInfoVersion={#FileVersion}
VersionInfoCompany=WorkBuddy
VersionInfoDescription=YOLO 数据标注工具箱安装器
VersionInfoProductName=YOLO 数据标注工具箱
VersionInfoProductVersion={#FileVersion}
VersionInfoCopyright=Copyright (C) 2026 WorkBuddy contributors

[Languages]
Name: "chinesesimp"; MessagesFile: "{#SourcePath}\languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\WorkBuddy\YOLO 数据标注工具箱"; Filename: "{app}\YOLO工具箱.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\YOLO 数据标注工具箱"; Filename: "{app}\YOLO工具箱.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\YOLO工具箱.exe"; Description: "启动 YOLO 数据标注工具箱"; Flags: nowait postinstall skipifsilent
