module.exports = {
  todayMacro: function(markdown) {
    return new Promise((resolve, reject)=> {
      var date = new Date();
      var year = date.getFullYear();
      var month = date.getMonth()+1;
      var day = date.getDate();
      markdown = markdown.replace(/\@today/gm, year + " 年 " + month + " 月 " + day + " 日 ");
      return resolve(markdown);
    })
  },
  variableMacro: function(markdown) {
    return new Promise((resolve, reject)=> {
      var variables = new Map();

      var def_re = /\(\s*!(\w+)\s*=\s*([亜-熙ぁ-んァ-ヶ\w]+)\s*\)/gm;
      // if (markdown.match(def_re)) {
      //   markdown = "true";
      // } else {
      //   markdown = "false";
      // }
      markdown = markdown.replace(def_re, ($0, $1, $2) => {variables.set($1, $2); return "";});

      var search_re = /\(!(\w+)\)/;
      // if (variables.size === 1) {
      //   markdown = "";
      //   variables.forEach((v,k) => markdown = markdown + k + " " + v + "\n");
      // } else {
      //   markdown = "false";
      // }
      // if (markdown.match(search_re)) {
      //   markdown = "true";
      // } else {
      //   markdown = "false";
      // }

      markdown = markdown.replace(search_re, ($0, $1) => {
        if (variables.has($1)) {
          return variables.get($1);
        } else {
          return $0;
        }
      });
      return resolve(markdown);
    })
  },
  onWillParseMarkdown: function(markdown) {
    return new Promise((resolve, reject)=> {
      // markdown = this.todayMacro(markdown);
      var date = new Date();
      var year = date.getFullYear();
      var month = date.getMonth()+1;
      var day = date.getDate();
      markdown = markdown.replace(/!today/gm, year + " 年 " + month + " 月 " + day + " 日 ");
      
      // markdown = this.variableMacro(markdown);
      var variables = new Map();
      var def_re = /\(\s*!(\w+)\s*=\s*([、。々〇〻\u3400-\u9FFF\uF900-\uFAFF\u3041-\u3096\u30A1-\u30FA\w\s]+)\s*\)/gm;
      markdown = markdown.replace(def_re, ($0, $1, $2) => {variables.set($1, $2); return "";});
      var search_re = /\(!(\w+)\)/;
      markdown = markdown.replace(search_re, ($0, $1) => {
        if (variables.has($1)) {
          return variables.get($1);
        } else {
          return $0;
        }
      });
      return resolve(markdown)
    })
  },
  onDidParseMarkdown: function(html, {cheerio}) {
    return new Promise((resolve, reject)=> {
      return resolve(html)
    })
  },
  onWillTransformMarkdown: function (markdown) {
        return new Promise((resolve, reject) => {
            return resolve(markdown);
        });
    },
  onDidTransformMarkdown: function (markdown) {
      return new Promise((resolve, reject) => {
          return resolve(markdown);
      });
  }
}