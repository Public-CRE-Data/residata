' ============================================================
' REIT Rental Data — GitHub CSV Importer
' Paste into VBA editor (Alt+F11 → Insert → Module)
' ============================================================
'
' SETUP:
'   1. Paste your fine-grained PAT into the TOKEN constant below
'   2. Run ImportREITData() or assign to a button
'   3. Pick which REIT to load from the dialog
'
' REQUIREMENTS: No external references needed — uses built-in
'   WinHttp.WinHttpRequest.5.1 and ADODB.Stream (standard on
'   all Windows / Office installations)
' ============================================================

Private Const GITHUB_TOKEN  As String = "PASTE_YOUR_FINE_GRAINED_PAT_HERE"
Private Const GITHUB_OWNER  As String = "Public-CRE-Data"
Private Const GITHUB_REPO   As String = "residata"
Private Const SCRAPE_DATE   As String = "2026-03-31"   ' update each week

' Available REIT files (add INVH once its run completes)
Private Const REIT_LIST As String = "MAA,CPT,EQR,AVB,UDR,ESS,INVH,ALL"


' ── Main entry point ─────────────────────────────────────────────────────────

Sub ImportREITData()
    Dim choice As String
    choice = InputBox( _
        "Enter REIT ticker to import (or ALL for every REIT):" & vbCrLf & vbCrLf & _
        "  MAA  CPT  EQR  AVB  UDR  ESS  INVH  ALL", _
        "REIT Rental Data Importer", "MAA")

    If choice = "" Then Exit Sub
    choice = UCase(Trim(choice))

    If choice = "ALL" Then
        Dim tickers As Variant
        tickers = Split("MAA,CPT,EQR,AVB,UDR,ESS,INVH", ",")
        Dim t As Variant
        For Each t In tickers
            ImportSingleREIT CStr(t)
        Next t
    Else
        ImportSingleREIT choice
    End If

    MsgBox "Import complete!", vbInformation, "REIT Data Importer"
End Sub


' ── Download and import a single REIT CSV ────────────────────────────────────

Private Sub ImportSingleREIT(ticker As String)
    Dim fileName As String
    fileName = LCase(ticker) & "_raw_" & SCRAPE_DATE & ".csv"

    Dim rawUrl As String
    rawUrl = "https://raw.githubusercontent.com/" & GITHUB_OWNER & "/" & _
             GITHUB_REPO & "/main/data/raw/" & fileName

    Application.StatusBar = "Downloading " & ticker & " data..."

    ' ── Download via WinHTTP ──────────────────────────────────────────────
    Dim http As Object
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")

    On Error GoTo DownloadError
    http.Open "GET", rawUrl, False
    http.SetRequestHeader "Authorization", "Bearer " & GITHUB_TOKEN
    http.SetRequestHeader "User-Agent", "Excel-VBA-ReitImporter"
    http.Send
    On Error GoTo 0

    If http.Status <> 200 Then
        MsgBox "HTTP " & http.Status & " for " & ticker & vbCrLf & _
               "Check token or that the file exists on GitHub.", _
               vbExclamation, "Download Failed"
        Exit Sub
    End If

    ' ── Save response to temp file ────────────────────────────────────────
    Dim tempPath As String
    tempPath = Environ("TEMP") & "\" & fileName

    Dim stream As Object
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 1  ' Binary
    stream.Open
    stream.Write http.ResponseBody
    stream.SaveToFile tempPath, 2  ' 2 = overwrite
    stream.Close

    ' ── Import CSV into a named sheet ─────────────────────────────────────
    Dim sheetName As String
    sheetName = ticker & "_" & Replace(SCRAPE_DATE, "-", "")

    ' Delete existing sheet with same name if present
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Sheets
        If ws.Name = sheetName Then
            Application.DisplayAlerts = False
            ws.Delete
            Application.DisplayAlerts = True
            Exit For
        End If
    Next ws

    ' Open the temp CSV and copy its sheet into this workbook
    Dim csvWb As Workbook
    Set csvWb = Workbooks.Open(tempPath, ReadOnly:=True)
    csvWb.Sheets(1).Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    csvWb.Close False

    ' Rename the newly copied sheet
    ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count).Name = sheetName

    ' ── Auto-format ───────────────────────────────────────────────────────
    Dim dataSheet As Worksheet
    Set dataSheet = ThisWorkbook.Sheets(sheetName)

    With dataSheet
        ' Freeze header row
        .Activate
        .Rows(2).Select
        ActiveWindow.FreezePanes = True
        .Range("A1").Select

        ' Bold header
        .Rows(1).Font.Bold = True

        ' AutoFit columns (cap at 30 chars wide)
        .Cells.EntireColumn.AutoFit
        Dim col As Range
        For Each col In .UsedRange.Columns
            If col.ColumnWidth > 30 Then col.ColumnWidth = 30
        Next col

        ' Format rent column as currency if present
        Dim rentCol As Range
        Set rentCol = .Rows(1).Find("rent", LookAt:=xlWhole)
        If Not rentCol Is Nothing Then
            .Columns(rentCol.Column).NumberFormat = "$#,##0"
        End If
    End With

    ' Clean up temp file
    On Error Resume Next
    Kill tempPath
    On Error GoTo 0

    Application.StatusBar = ticker & " loaded → sheet: " & sheetName
    Exit Sub

DownloadError:
    MsgBox "Network error downloading " & ticker & vbCrLf & Err.Description, _
           vbCritical, "Download Error"
    Application.StatusBar = False
End Sub


' ── Optional: import unit registry ───────────────────────────────────────────

Sub ImportUnitRegistry()
    Dim rawUrl As String
    rawUrl = "https://raw.githubusercontent.com/" & GITHUB_OWNER & "/" & _
             GITHUB_REPO & "/main/data/registry/unit_registry.csv"

    Dim http As Object
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")
    http.Open "GET", rawUrl, False
    http.SetRequestHeader "Authorization", "Bearer " & GITHUB_TOKEN
    http.SetRequestHeader "User-Agent", "Excel-VBA-ReitImporter"
    http.Send

    If http.Status <> 200 Then
        MsgBox "HTTP " & http.Status & " — check token.", vbExclamation
        Exit Sub
    End If

    Dim tempPath As String
    tempPath = Environ("TEMP") & "\unit_registry.csv"

    Dim stream As Object
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 1
    stream.Open
    stream.Write http.ResponseBody
    stream.SaveToFile tempPath, 2
    stream.Close

    ' Remove old sheet
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Sheets
        If ws.Name = "Unit_Registry" Then
            Application.DisplayAlerts = False
            ws.Delete
            Application.DisplayAlerts = True
            Exit For
        End If
    Next ws

    Dim csvWb As Workbook
    Set csvWb = Workbooks.Open(tempPath, ReadOnly:=True)
    csvWb.Sheets(1).Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    csvWb.Close False
    ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count).Name = "Unit_Registry"

    On Error Resume Next
    Kill tempPath
    On Error GoTo 0

    MsgBox "Unit registry imported.", vbInformation
End Sub
