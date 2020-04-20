#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

// required to decode UTF-8 encoded files
#include <codecvt>
#include <locale>
#include <cwctype>

#define MAX(a, b) ((a)<(b)?(b):(a))

int longest_common_subsequence(std::wstring& test_seq, 
                               std::wstring& source_seq)
{
  std::vector<std::vector<unsigned int>> 
  lcs_array(test_seq.length()+1u, std::vector<unsigned int>(source_seq.length()+1u, 0u));

  for (unsigned int row_iter = 1u; row_iter < lcs_array.size(); row_iter++)
  {
    for (unsigned int col_iter = 1u; col_iter < lcs_array[0].size(); col_iter++)
    {
      if (_wcsicmp(&source_seq[row_iter-1u], &test_seq[col_iter-1u]) < 0)
      {
        lcs_array[row_iter][col_iter] = 1u + lcs_array[row_iter-1][col_iter-1];
      }
      else
      {
        lcs_array[row_iter][col_iter] = MAX(lcs_array[row_iter-1u][col_iter], 
                                            lcs_array[row_iter][col_iter-1u]);
      }
    }
  }
  return (lcs_array.back()).back();
}


int main(int argc, char** argv)
{
  std::string source_filename, test_filename;
  if (argc == 3)
  {
    test_filename   = std::string{argv[1]};
    source_filename = std::string{argv[2]};

    // define string stream to hold file contents
    std::wstringstream test_file_contents;
    std::wstringstream source_file_contents;

    std::wifstream test_fileobj, source_fileobj;
    test_fileobj.open(test_filename, std::ios_base::in);
    test_fileobj.imbue(std::locale("en_US.UTF-8"));

    if (test_fileobj.is_open())
    {
      std::wstring temp_str;
      while (!test_fileobj.eof())
      {
        std::getline(test_fileobj, temp_str);
        test_file_contents << temp_str;
      }
    }
    test_fileobj.close();

    source_fileobj.open(source_filename, std::ios_base::in);
    source_fileobj.imbue(std::locale("en_US.UTF-8"));
    if (source_fileobj.is_open())
    {
      std::wstring temp_str;
      while (!source_fileobj.eof())
      {
        std::getline(source_fileobj, temp_str);
        source_file_contents << temp_str;
      }
    }
    source_fileobj.close();

    // std::wcout.imbue(std::locale("en_US.UTF-8"));
    // std::wcout << test_file_contents.str() << '\n';
    // std::cout << "--------------------------\n" ;
    // std::wcout << source_file_contents.str() << '\n';
    
    int lcs_result = longest_common_subsequence(
                                                source_file_contents.str(),
                                                test_file_contents.str()
                                                );
    std::cout << lcs_result;
  }
  return EXIT_SUCCESS;
}